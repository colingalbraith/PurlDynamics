"""Offset Geometric Contact — core geometry and mesh-level contact.

[Chen et al., SIGGRAPH 2025]

Penetration-free contact by offsetting geometry by radius delta.
Contact normals are geometrically well-defined (direction from closest
point to query point), avoiding IPC's stiffness issues. Per-vertex
displacement bounds guarantee no new penetrations per step.

For sphere-level contact used by the chain demo, see contact.py.
For full mesh-level contact on deformable triangles, see MeshContact.
"""

import taichi as ti


@ti.func
def point_triangle_closest_point(p: ti.template(), a: ti.template(),
                                  b: ti.template(), c: ti.template()):
    """Closest point on triangle (a,b,c) to point p.
    Returns (distance, closest_point). Voronoi region test."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab.dot(ap), ac.dot(ap)

    closest = a
    if d1 <= 0.0 and d2 <= 0.0:
        closest = a
    else:
        bp = p - b
        d3, d4 = ab.dot(bp), ac.dot(bp)
        if d3 >= 0.0 and d4 <= d3:
            closest = b
        else:
            cp = p - c
            d5, d6 = ab.dot(cp), ac.dot(cp)
            if d6 >= 0.0 and d5 <= d6:
                closest = c
            else:
                vc = d1 * d4 - d3 * d2
                if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                    closest = a + d1 / (d1 - d3) * ab
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        closest = a + d2 / (d2 - d6) * ac
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            closest = b + w * (c - b)
                        else:
                            denom = 1.0 / (va + vb + vc)
                            closest = a + vb * denom * ab + vc * denom * ac

    return (p - closest).norm(), closest


@ti.data_oriented
class MeshContact:
    """Full vertex-face OGC for deformable triangle meshes.

    Brute-force O(V*F). For large meshes, add a spatial hash broadphase.
    Object IDs skip intra-object contacts. Set all to 0 for self-collision.
    """

    def __init__(self, max_verts, max_faces, delta=0.01):
        self.delta = delta

        self.verts = ti.Vector.field(3, dtype=ti.f64, shape=max_verts)
        self.face_idx = ti.Vector.field(3, dtype=ti.i32, shape=max_faces)
        self.n_verts = ti.field(dtype=ti.i32, shape=())
        self.n_faces = ti.field(dtype=ti.i32, shape=())

        self.vert_obj_id = ti.field(dtype=ti.i32, shape=max_verts)
        self.face_obj_id = ti.field(dtype=ti.i32, shape=max_faces)

        self.correction = ti.Vector.field(3, dtype=ti.f64, shape=max_verts)
        self.contact_count = ti.field(dtype=ti.i32, shape=max_verts)
        self.disp_bound = ti.field(dtype=ti.f64, shape=max_verts)
        self.total_contacts = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def compute_displacement_bounds(self):
        """Per-vertex bounds: (dist_to_face - delta) / 2 for nearby faces."""
        nv, nf, delta = self.n_verts[None], self.n_faces[None], self.delta

        for i in range(nv):
            self.disp_bound[i] = 1e10

        for i in range(nv):
            for f in range(nf):
                if self.vert_obj_id[i] == self.face_obj_id[f]:
                    continue
                i0, i1, i2 = self.face_idx[f][0], self.face_idx[f][1], self.face_idx[f][2]
                dist, _ = point_triangle_closest_point(
                    self.verts[i], self.verts[i0], self.verts[i1], self.verts[i2])
                if delta < dist < 3.0 * delta:
                    ti.atomic_min(self.disp_bound[i], (dist - delta) * 0.5)

    @ti.kernel
    def detect_and_resolve(self):
        """Find contacts within offset delta, compute position corrections."""
        nv, nf, delta = self.n_verts[None], self.n_faces[None], self.delta
        self.total_contacts[None] = 0

        for i in range(nv):
            self.correction[i] = ti.Vector([0.0, 0.0, 0.0])
            self.contact_count[i] = 0

        for i in range(nv):
            for f in range(nf):
                if self.vert_obj_id[i] == self.face_obj_id[f]:
                    continue
                i0, i1, i2 = self.face_idx[f][0], self.face_idx[f][1], self.face_idx[f][2]
                if i == i0 or i == i1 or i == i2:
                    continue
                dist, closest = point_triangle_closest_point(
                    self.verts[i], self.verts[i0], self.verts[i1], self.verts[i2])
                if dist < delta and dist > 1e-10:
                    normal = (self.verts[i] - closest) / dist
                    ti.atomic_add(self.correction[i], (delta - dist) * normal)
                    ti.atomic_add(self.contact_count[i], 1)
                    ti.atomic_add(self.total_contacts[None], 1)

    @ti.kernel
    def apply_corrections(self):
        for i in range(self.n_verts[None]):
            if self.contact_count[i] > 0:
                self.verts[i] += self.correction[i] / ti.cast(self.contact_count[i], ti.f64)

    @ti.kernel
    def clamp_displacements(self, prev_verts: ti.template()):
        for i in range(self.n_verts[None]):
            disp = self.verts[i] - prev_verts[i]
            disp_len = disp.norm()
            if disp_len > self.disp_bound[i] and disp_len > 1e-12:
                self.verts[i] = prev_verts[i] + disp * (self.disp_bound[i] / disp_len)
