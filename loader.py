"""Asset loading for the chain mail simulation.

Parses link.obj mesh and chains.json configuration, builds the
connectivity graph and per-link adjacency for orientation tracking.
"""

import numpy as np
import json
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def load_obj(path):
    """Parse OBJ file, triangulate quads. Returns (vertices, faces)."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v' and len(parts) >= 4:
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                idx = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def euler_xyz_to_matrix(e):
    """Euler XYZ (degrees) -> 3x3 rotation matrix."""
    rx, ry, rz = np.radians(e)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @
            np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @
            np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def compute_vertex_normals(verts, faces):
    """Per-vertex normals by area-weighted face normal averaging."""
    normals = np.zeros_like(verts)
    for f in faces:
        e1 = verts[f[1]] - verts[f[0]]
        e2 = verts[f[2]] - verts[f[0]]
        fn = np.cross(e1, e2)
        normals[f[0]] += fn
        normals[f[1]] += fn
        normals[f[2]] += fn
    norms = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    return (normals / norms).astype(np.float32)


def load_chain_scene(conn_threshold=1.2, max_neighbors=8):
    """Load assets and build scene data.

    Returns dict with numpy arrays for simulation and rendering.
    conn_threshold: max distance to consider two links connected.
                    1.2 captures both direct (0.8) and diagonal (1.149) links.
    """
    link_verts, link_faces = load_obj(os.path.join(ASSETS_DIR, "link.obj"))
    base_normals = compute_vertex_normals(link_verts, link_faces)
    NV, NF = len(link_verts), len(link_faces)

    with open(os.path.join(ASSETS_DIR, "chains.json")) as f:
        chain_data = json.load(f)

    N = len(chain_data)
    init_pos = np.array([d['p'] for d in chain_data], dtype=np.float32)
    init_euler = np.array([d['euler'] for d in chain_data], dtype=np.float32)
    init_mass = np.array([d['mass'] for d in chain_data], dtype=np.float32)

    # Pre-rotate mesh per link's initial orientation
    local_verts = np.zeros((N, NV, 3), dtype=np.float32)
    local_norms = np.zeros((N, NV, 3), dtype=np.float32)
    for i in range(N):
        R = euler_xyz_to_matrix(init_euler[i])
        local_verts[i] = (R @ link_verts.T).T
        local_norms[i] = (R @ base_normals.T).T

    # Connectivity graph
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(init_pos[i] - init_pos[j]))
            if d < conn_threshold:
                edges.append((i, j, d))
    N_EDGES = len(edges)
    edge_rest = np.array([e[2] for e in edges], dtype=np.float32)

    # Per-link adjacency (for SVD orientation tracking)
    nb_idx = np.full((N, max_neighbors), -1, dtype=np.int32)
    nb_count = np.zeros(N, dtype=np.int32)
    nb_rel = np.zeros((N, max_neighbors, 3), dtype=np.float32)
    for i, j, d in edges:
        if nb_count[i] < max_neighbors:
            m = nb_count[i]
            nb_idx[i, m] = j
            nb_rel[i, m] = init_pos[j] - init_pos[i]
            nb_count[i] += 1
        if nb_count[j] < max_neighbors:
            m = nb_count[j]
            nb_idx[j, m] = i
            nb_rel[j, m] = init_pos[i] - init_pos[j]
            nb_count[j] += 1

    return {
        "link_verts": link_verts, "link_faces": link_faces,
        "base_normals": base_normals,
        "init_pos": init_pos, "init_euler": init_euler, "init_mass": init_mass,
        "local_verts": local_verts, "local_norms": local_norms,
        "edges": edges, "edge_rest": edge_rest,
        "nb_idx": nb_idx, "nb_count": nb_count, "nb_rel": nb_rel,
        "N": N, "NV": NV, "NF": NF, "N_EDGES": N_EDGES,
        "MAX_NB": max_neighbors,
    }
