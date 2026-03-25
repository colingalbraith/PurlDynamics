"""3D Chain Mail — Projective Dynamics + Offset Geometric Contact

144 interlocking chain links simulated with Projective Dynamics
[Bouaziz et al. 2014] and OGC contact [Chen et al., SIGGRAPH 2025].

The PD system matrix is prefactored once — each global solve is a
single matrix-vector multiply, giving real-time performance.

Controls:
  R        Reset
  Space    Pause (starts paused)
  G        Cycle gravity direction
  +/-      Stiffness
  D        Toggle damping
  Mouse    Orbit (LMB), Zoom (scroll)

Run:  cd chains && python main.py
"""

import taichi as ti
import numpy as np

ti.init(arch=ti.vulkan)

from loader import load_chain_scene
from pd import ChainPD

# --- Scene ---

scene_data = load_chain_scene()
N, NV, NF = scene_data["N"], scene_data["NV"], scene_data["NF"]
TOTAL_V, TOTAL_F = N * NV, N * NF

solver = ChainPD(scene_data, stiffness=5000.0, damping=0.01,
                 pd_iters=3, ogc_delta=0.25, max_stretch=1.3)

print(f"Links: {N} ({(scene_data['init_mass'] < 0).sum()} pinned)  "
      f"Edges: {scene_data['N_EDGES']}  "
      f"Mesh: {TOTAL_V} verts, {TOTAL_F} tris")

# --- Rendering ---

mesh_verts = ti.Vector.field(3, ti.f32, shape=TOTAL_V)
mesh_normals = ti.Vector.field(3, ti.f32, shape=TOTAL_V)
mesh_colors = ti.Vector.field(3, ti.f32, shape=TOTAL_V)
mesh_indices = ti.field(ti.i32, shape=TOTAL_F * 3)


def init_rendering():
    link_faces = scene_data["link_faces"]
    idx_np = np.zeros(TOTAL_F * 3, dtype=np.int32)
    for i in range(N):
        idx_np[i * NF * 3:(i + 1) * NF * 3] = (link_faces + i * NV).reshape(-1)
    mesh_indices.from_numpy(idx_np)

    colors_np = np.zeros((TOTAL_V, 3), dtype=np.float32)
    euler, mass = scene_data["init_euler"], scene_data["init_mass"]
    for i in range(N):
        e = euler[i]
        if mass[i] < 0:
            c = [0.25, 0.35, 0.65]
        elif abs(e[0] - 90) < 1 and abs(e[2] - 90) < 1:
            c = [0.85, 0.55, 0.35]
        elif abs(e[2] - 90) < 1:
            c = [0.75, 0.75, 0.78]
        else:
            c = [0.6, 0.65, 0.55]
        colors_np[i * NV:(i + 1) * NV] = c
    mesh_colors.from_numpy(colors_np)


@ti.kernel
def update_mesh(pos: ti.template(), rot: ti.template(),
                lv: ti.template(), ln: ti.template()):
    for i, j in ti.ndrange(N, NV):
        idx = i * NV + j
        R = rot[i]
        mesh_verts[idx] = R @ lv[i, j] + pos[i]
        mesh_normals[idx] = R @ ln[i, j]


# --- Main loop ---

solver.reset()
init_rendering()
update_mesh(solver.pos, solver.link_rot, solver.local_v, solver.local_n)

window = ti.ui.Window("Chain Mail — PD + OGC", (1280, 720), vsync=True)
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()

center = scene_data["init_pos"].mean(axis=0)
camera.position(center[0], center[1] - 15.0, center[2] + 10.0)
camera.lookat(center[0], center[1], center[2])
camera.up(0, 1, 0)
camera.fov(50)

paused = True
grav_idx = 0
GRAVS = [[0, -9.81, 0], [0, 9.81, 0], [-9.81, 0, 0],
         [9.81, 0, 0], [0, 0, -9.81], [0, 0, 9.81]]

DT, SUB_STEPS = 0.004, 4

print("Controls: Space=pause  R=reset  G=gravity  +/-=stiffness  D=damping")

while window.running:
    for e in window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.ESCAPE:
            window.running = False
        elif e.key == 'r':
            solver.reset()
        elif e.key == ti.ui.SPACE:
            paused = not paused
            print("PAUSED" if paused else "RUNNING")
        elif e.key == 'g':
            grav_idx = (grav_idx + 1) % len(GRAVS)
            solver.gravity[None] = ti.Vector(GRAVS[grav_idx])
        elif e.key in ('=', ']'):
            solver.stiffness *= 2.0
            print(f"Stiffness: {solver.stiffness:.0f}")
        elif e.key in ('-', '['):
            solver.stiffness = max(100.0, solver.stiffness / 2.0)
            print(f"Stiffness: {solver.stiffness:.0f}")
        elif e.key == 'd':
            solver.damping = 0.0 if solver.damping > 0.005 else 0.01

    camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.LMB)

    if not paused:
        solver.step(DT, SUB_STEPS)

    solver.compute_rotations()
    update_mesh(solver.pos, solver.link_rot, solver.local_v, solver.local_n)

    scene.set_camera(camera)
    scene.ambient_light((0.35, 0.35, 0.4))
    scene.point_light(pos=(center[0], center[1] - 12, center[2] + 8),
                      color=(1.0, 0.95, 0.85))
    scene.point_light(pos=(center[0] + 10, center[1] - 5, center[2] - 5),
                      color=(0.3, 0.35, 0.45))
    scene.mesh(mesh_verts, indices=mesh_indices, normals=mesh_normals,
               per_vertex_color=mesh_colors)
    canvas.scene(scene)
    window.show()
