"""Projective Dynamics solver for 3D chain mail.

[Bouaziz et al. 2014] — alternates local spring projections with a
global linear solve. The system matrix (M/h^2 + k*L) is constant for
fixed topology, so we precompute its inverse once and store it as a
Taichi field — the global solve runs entirely on GPU as a single
matrix-vector multiply kernel. No CPU round-trips during simulation.
"""

import taichi as ti
import numpy as np

from contact import resolve_and_clamp, clamp_max_stretch
from orientation import compute_rotations as _compute_rotations


@ti.data_oriented
class ChainPD:

    def __init__(self, scene_data, stiffness=5000.0, damping=0.01,
                 pd_iters=3, ogc_delta=0.25, max_stretch=1.3):
        self.N = scene_data["N"]
        self.NV = scene_data["NV"]
        self.NF = scene_data["NF"]
        self.N_EDGES = scene_data["N_EDGES"]
        self.MAX_NB = scene_data["MAX_NB"]
        self.DOF = 3 * self.N

        self._stiffness = stiffness
        self.damping = damping
        self.pd_iters = pd_iters
        self.ogc_delta = ogc_delta
        self.max_stretch = max_stretch

        N, NE, NV, MNB = self.N, self.N_EDGES, self.NV, self.MAX_NB

        # Simulation state
        self.pos = ti.Vector.field(3, ti.f32, shape=N)
        self.vel = ti.Vector.field(3, ti.f32, shape=N)
        self.pos_old = ti.Vector.field(3, ti.f32, shape=N)
        self.mass = ti.field(ti.f32, shape=N)
        self.inv_mass = ti.field(ti.f32, shape=N)
        self.gravity = ti.Vector.field(3, ti.f32, shape=())

        # Edge constraints
        self.edge_a = ti.field(ti.i32, shape=NE)
        self.edge_b = ti.field(ti.i32, shape=NE)
        self.edge_rest = ti.field(ti.f32, shape=NE)

        # PD workspace
        self.s_n = ti.Vector.field(3, ti.f32, shape=N)
        self.rhs = ti.Vector.field(3, ti.f32, shape=N)

        # Prefactored inverse (GPU-resident for zero-copy solves)
        self.A_inv = ti.field(ti.f32, shape=(self.DOF, self.DOF))

        # Contact workspace
        self.disp_bound = ti.field(ti.f32, shape=N)
        self.pos_pre_solve = ti.Vector.field(3, ti.f32, shape=N)

        # Orientation tracking
        self.nb_idx = ti.field(ti.i32, shape=(N, MNB))
        self.nb_count = ti.field(ti.i32, shape=N)
        self.nb_rel = ti.Vector.field(3, ti.f32, shape=(N, MNB))
        self.link_rot = ti.Matrix.field(3, 3, ti.f32, shape=N)

        # Per-link mesh (for rendering)
        self.local_v = ti.Vector.field(3, ti.f32, shape=(N, NV))
        self.local_n = ti.Vector.field(3, ti.f32, shape=(N, NV))

        # Numpy data for reset
        self._init_pos = scene_data["init_pos"].copy()
        self._init_vel = np.zeros((N, 3), dtype=np.float32)

        # Build system matrix components and load constant data
        inv_mass_np = np.array(
            [0.0 if m < 0 else 1.0 / m for m in scene_data["init_mass"]],
            dtype=np.float32)
        mass_np = np.abs(scene_data["init_mass"]).astype(np.float32)
        self._build_laplacian(scene_data["edges"], inv_mass_np,
                              np.abs(scene_data["init_mass"]).astype(np.float64))
        self._init_fields(scene_data, mass_np, inv_mass_np)

        self._cached_dt = None
        self._cached_stiffness = None

    @property
    def stiffness(self):
        return self._stiffness

    @stiffness.setter
    def stiffness(self, val):
        self._stiffness = val
        self._cached_dt = None  # force re-prefactor

    # -----------------------------------------------------------------
    #  Prefactoring
    # -----------------------------------------------------------------

    def _build_laplacian(self, edges, inv_mass_np, mass_np):
        """Build mass diagonal and graph Laplacian as numpy arrays."""
        N = self.N
        self._M_diag = np.zeros(3 * N, dtype=np.float64)
        for i in range(N):
            m = -1.0 if inv_mass_np[i] < 1e-8 else mass_np[i]
            for d in range(3):
                self._M_diag[3 * i + d] = m

        self._L = np.zeros((3 * N, 3 * N), dtype=np.float64)
        for a_idx, b_idx, _ in edges:
            for d in range(3):
                if inv_mass_np[a_idx] >= 1e-8:
                    self._L[3 * a_idx + d, 3 * a_idx + d] += 1.0
                    self._L[3 * a_idx + d, 3 * b_idx + d] -= 1.0
                if inv_mass_np[b_idx] >= 1e-8:
                    self._L[3 * b_idx + d, 3 * b_idx + d] += 1.0
                    self._L[3 * b_idx + d, 3 * a_idx + d] -= 1.0

    def _prefactor(self, dt):
        """Compute inv(M/h^2 + k*L) and upload to GPU. Cached."""
        if self._cached_dt == dt and self._cached_stiffness == self._stiffness:
            return

        A = self._stiffness * self._L.copy()
        inv_h2 = 1.0 / (dt * dt)
        for i in range(3 * self.N):
            if self._M_diag[i] < 0:
                A[i, :] = 0.0
                A[i, i] = 1.0
            else:
                A[i, i] += self._M_diag[i] * inv_h2

        self.A_inv.from_numpy(np.linalg.inv(A).astype(np.float32))
        self._cached_dt = dt
        self._cached_stiffness = self._stiffness

    @ti.kernel
    def _direct_solve(self):
        """Global solve on GPU: pos = A_inv @ rhs. No CPU round-trip."""
        for i in range(self.N):
            result = ti.Vector([0.0, 0.0, 0.0])
            for d in ti.static(range(3)):
                row = 3 * i + d
                val = 0.0
                for j in range(self.N):
                    for e in ti.static(range(3)):
                        val += self.A_inv[row, 3 * j + e] * self.rhs[j][e]
                result[d] = val
            self.pos[i] = result

    # -----------------------------------------------------------------
    #  Init / reset
    # -----------------------------------------------------------------

    def _init_fields(self, sd, mass_np, inv_mass_np):
        """Load constant data using from_numpy (bulk upload, no Python loops)."""
        self.edge_a.from_numpy(np.array([e[0] for e in sd["edges"]], dtype=np.int32))
        self.edge_b.from_numpy(np.array([e[1] for e in sd["edges"]], dtype=np.int32))
        self.edge_rest.from_numpy(sd["edge_rest"])
        self.mass.from_numpy(mass_np)
        self.inv_mass.from_numpy(inv_mass_np)
        self.nb_idx.from_numpy(sd["nb_idx"])
        self.nb_count.from_numpy(sd["nb_count"])
        self.nb_rel.from_numpy(sd["nb_rel"].astype(np.float32))
        self.local_v.from_numpy(sd["local_verts"])
        self.local_n.from_numpy(sd["local_norms"])

    def reset(self):
        self.pos.from_numpy(self._init_pos)
        self.vel.from_numpy(self._init_vel)
        self.gravity[None] = ti.Vector([0.0, -9.81, 0.0])
        self._reset_rotations()

    @ti.kernel
    def _reset_rotations(self):
        for i in range(self.N):
            self.link_rot[i] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # -----------------------------------------------------------------
    #  Timestep
    # -----------------------------------------------------------------

    def step(self, dt, sub_steps=4):
        self._prefactor(dt)
        for _ in range(sub_steps):
            self._sub_step(dt)

    def _sub_step(self, dt):
        self._predict_and_init(dt)
        for _ in range(self.pd_iters):
            self._project_and_build_rhs(dt)
            self._direct_solve()
        resolve_and_clamp(self.pos, self.inv_mass, self.pos_pre_solve,
                          self.disp_bound, self.ogc_delta, self.N)
        clamp_max_stretch(self.pos, self.inv_mass, self.edge_a, self.edge_b,
                          self.edge_rest, self.max_stretch, self.N_EDGES)
        self._update_velocity(dt)

    def compute_rotations(self):
        _compute_rotations(self.pos, self.link_rot, self.nb_idx,
                           self.nb_count, self.nb_rel, self.N, self.MAX_NB)

    # -----------------------------------------------------------------
    #  PD kernels
    # -----------------------------------------------------------------

    @ti.kernel
    def _predict_and_init(self, dt: ti.f32):
        for i in range(self.N):
            self.pos_old[i] = self.pos[i]
            if self.inv_mass[i] > 0.0:
                self.s_n[i] = self.pos[i] + dt * self.vel[i] + dt * dt * self.gravity[None]
                self.pos[i] = self.s_n[i]
            else:
                self.s_n[i] = self.pos[i]

    @ti.kernel
    def _project_and_build_rhs(self, dt: ti.f32):
        """Fused local projection + RHS assembly. Uses atomic_add for
        thread-safe edge contributions (multiple edges per vertex)."""
        inv_h2 = 1.0 / (dt * dt)
        k = self._stiffness

        for i in range(self.N):
            if self.inv_mass[i] < 1e-8:
                self.rhs[i] = self.pos[i]
            else:
                self.rhs[i] = self.mass[i] * inv_h2 * self.s_n[i]

        for c in range(self.N_EDGES):
            diff = self.pos[self.edge_b[c]] - self.pos[self.edge_a[c]]
            dist = diff.norm()
            p = ti.Vector([0.0, 0.0, 0.0])
            if dist < 1e-8:
                p = ti.Vector([self.edge_rest[c], 0.0, 0.0])
            else:
                p = self.edge_rest[c] * diff / dist

            i, j = self.edge_a[c], self.edge_b[c]
            if self.inv_mass[i] >= 1e-8:
                ti.atomic_add(self.rhs[i], -k * p)
            if self.inv_mass[j] >= 1e-8:
                ti.atomic_add(self.rhs[j], k * p)

    @ti.kernel
    def _update_velocity(self, dt: ti.f32):
        for i in range(self.N):
            self.vel[i] = (1.0 - self.damping) * (self.pos[i] - self.pos_old[i]) / dt
