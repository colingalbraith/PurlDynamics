"""Sphere-level OGC contact and max-stretch clamping.

Applies OGC [Chen et al., SIGGRAPH 2025] principles at the rigid body
center level: each link is a sphere of radius delta. Displacement bounds
guarantee no new contacts per step.

For the full mesh-level vertex-face algorithm, see ogc.py.
"""

import taichi as ti


@ti.kernel
def resolve_and_clamp(pos: ti.template(), inv_mass: ti.template(),
                      pos_pre_solve: ti.template(), disp_bound: ti.template(),
                      delta: ti.f32, N: ti.i32):
    """Sphere contact with OGC displacement bounds.

    Pairs closer than 2*delta get pushed apart. Bounds of
    (dist - 2*delta) / 2 guarantee no new contacts per step.
    """
    min_dist = 2.0 * delta

    for i in range(N):
        pos_pre_solve[i] = pos[i]
        disp_bound[i] = 1e10

    for i, j in ti.ndrange(N, N):
        if j <= i:
            continue
        w = inv_mass[i] + inv_mass[j]
        if w < 1e-8:
            continue
        d = pos[i] - pos[j]
        dist = d.norm()

        if dist > min_dist and dist < 6.0 * delta:
            bound = (dist - min_dist) * 0.5
            if inv_mass[i] > 0.0:
                ti.atomic_min(disp_bound[i], bound)
            if inv_mass[j] > 0.0:
                ti.atomic_min(disp_bound[j], bound)

        if dist < min_dist and dist > 1e-6:
            n = d / dist
            depth = min_dist - dist
            if inv_mass[i] > 0.0:
                ti.atomic_add(pos[i], inv_mass[i] / w * depth * n)
            if inv_mass[j] > 0.0:
                ti.atomic_add(pos[j], -inv_mass[j] / w * depth * n)

    for i in range(N):
        if inv_mass[i] < 1e-8:
            continue
        disp = pos[i] - pos_pre_solve[i]
        disp_len = disp.norm()
        if disp_len > disp_bound[i] and disp_len > 1e-8:
            pos[i] = pos_pre_solve[i] + disp * (disp_bound[i] / disp_len)


@ti.kernel
def clamp_max_stretch(pos: ti.template(), inv_mass: ti.template(),
                      edge_a: ti.template(), edge_b: ti.template(),
                      edge_rest: ti.template(),
                      max_ratio: ti.f32, N_EDGES: ti.i32):
    """Prevent connected elements from separating beyond max_ratio * rest_length."""
    for c in range(N_EDGES):
        i, j = edge_a[c], edge_b[c]
        d = pos[i] - pos[j]
        dist = d.norm()
        max_dist = max_ratio * edge_rest[c]
        if dist > max_dist and dist > 1e-6:
            n = d / dist
            C = dist - max_dist
            w = inv_mass[i] + inv_mass[j]
            if w < 1e-6:
                continue
            corr = C / w * n
            if inv_mass[i] > 0.0:
                pos[i] -= inv_mass[i] * corr
            if inv_mass[j] > 0.0:
                pos[j] += inv_mass[j] * corr
