"""SVD-based orientation tracking using the Kabsch algorithm.

Derives per-link rotation matrices from how neighbor positions
have moved relative to their initial configuration.
"""

import taichi as ti


@ti.kernel
def compute_rotations(pos: ti.template(), link_rot: ti.template(),
                      nb_idx: ti.template(), nb_count: ti.template(),
                      nb_rel: ti.template(), N: ti.i32, MAX_NB: ti.i32):
    for i in range(N):
        H = ti.Matrix.zero(ti.f32, 3, 3)
        nc = nb_count[i]
        for m in range(MAX_NB):
            if m < nc:
                j = nb_idx[i, m]
                H += (pos[j] - pos[i]).outer_product(nb_rel[i, m])
        if nc > 0:
            U, sig, V = ti.svd(H)
            R = U @ V.transpose()
            if R.determinant() < 0.0:
                for p in ti.static(range(3)):
                    U[p, 2] *= -1.0
                R = U @ V.transpose()
            link_rot[i] = R
        else:
            link_rot[i] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
