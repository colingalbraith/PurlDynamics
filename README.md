# PurlDynamics

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Taichi](https://img.shields.io/badge/taichi-1.7.4-orange.svg)](https://github.com/taichi-dev/taichi)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Real-time chain mail simulation using **Projective Dynamics** and **Offset Geometric Contact** on the GPU.

<!-- Add screenshots/gifs here -->
<!-- ![Demo](docs/images/demo.gif) -->

<p align="center">
  <img src="docs/images/placeholder.png" alt="Chain mail simulation" width="720">
</p>

---

## Overview

PurlDynamics simulates 144 interlocking chain links as a curtain pinned at the boundary, dropping under gravity. The solver uses Projective Dynamics with a prefactored system matrix for real-time performance and OGC-style contact for penetration prevention.

### Methods

| Method | Paper | Role |
|--------|-------|------|
| Projective Dynamics | [Bouaziz et al. 2014](https://doi.org/10.1111/cgf.12346) | Implicit time integration via local/global alternation |
| Offset Geometric Contact | [Chen et al., SIGGRAPH 2025](https://graphics.cs.utah.edu/research/projects/ogc/) | Penetration-free contact with displacement bounds |
| Kabsch Algorithm | [Kabsch 1976](https://doi.org/10.1107/S0567739476001873) | SVD-based per-link orientation tracking |

### Key Features

- **Prefactored direct solve** — system matrix inverted once, each global step is a single GPU matmul
- **OGC displacement bounds** — per-link bounds guarantee no new contacts per step
- **Sphere-level + mesh-level OGC** — practical sphere contact for the demo, full vertex-face module for future use
- **SVD orientation tracking** — link meshes rotate correctly as the chain deforms
- **~270 FPS** simulation on Apple Silicon (CPU), higher on discrete GPU

---

## Project Structure

```
PurlDynamics/
├── main.py              # GGUI demo — rendering, camera, controls
├── pd.py                # Projective Dynamics solver (prefactored global solve)
├── ogc.py               # Core OGC geometry + full mesh-level MeshContact
├── contact.py           # Sphere-level OGC contact + max-stretch clamping
├── orientation.py       # SVD Kabsch rotation tracking
├── loader.py            # OBJ parser, scene loader, connectivity builder
├── assets/
│   ├── link.obj         # Chain link mesh (176 verts, 352 tris)
│   ├── link.ma          # Maya source file
│   └── chains.json      # 144 link positions, orientations, masses
└── docs/
    └── images/          # Screenshots and diagrams
```

---

## Getting Started

### Requirements

- Python 3.12
- [Taichi](https://github.com/taichi-dev/taichi) 1.7+
- NumPy

### Install

```bash
git clone https://github.com/colingalbraith/PurlDynamics.git
cd PurlDynamics
python -m venv .venv
source .venv/bin/activate
pip install taichi numpy
```

### Run

```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| `Space` | Pause / unpause (starts paused) |
| `R` | Reset simulation |
| `G` | Cycle gravity direction |
| `+` / `-` | Increase / decrease stiffness |
| `D` | Toggle damping |
| `LMB drag` | Orbit camera |
| `Scroll` | Zoom |
| `Esc` | Quit |

---

## Architecture

```
main.py
  └── ChainPD (pd.py)
        ├── _prefactor()            # inv(M/h² + k·L) → GPU field (once)
        ├── _predict_and_init()     # inertial prediction s_n
        ├── _project_and_build_rhs()# local step + RHS assembly
        ├── _direct_solve()         # GPU matmul: pos = A_inv @ rhs
        ├── resolve_and_clamp()     # OGC sphere contact (contact.py)
        ├── clamp_max_stretch()     # max-stretch safety (contact.py)
        ├── _update_velocity()      # damped velocity from position delta
        └── compute_rotations()     # SVD Kabsch (orientation.py)
```

The system matrix `A = M/h² + k·L` is constant for fixed topology, stiffness, and timestep. It is inverted once in NumPy and stored as a Taichi GPU field. Each global solve is a single 432×432 matrix-vector multiply kernel — no iterative solver, no CPU round-trips during simulation.

---

## Extending

**Different chain patterns:** Replace `assets/chains.json` with new link positions/orientations. The connectivity threshold in `loader.py` (default 1.2) may need adjusting for different link spacings.

**Different link meshes:** Replace `assets/link.obj`. If using multiple mesh types, extend `loader.py` to support per-link mesh references via the `obj` field in `chains.json`.

**Full mesh-level OGC:** `ogc.py` contains `MeshContact` with brute-force vertex-face contact. For production use, add a spatial hash broadphase to cull distant pairs.

---

## References

1. S. Bouaziz, S. Martin, T. Liu, L. Kavan, M. Pauly. *Projective Dynamics: Fusing Constraint Projections for Fast Simulation.* ACM SIGGRAPH 2014.
2. A. H. Chen, J. Hsu, Z. Liu, M. Macklin, Y. Yang, C. Yuksel. *Offset Geometric Contact.* ACM SIGGRAPH 2025.
3. W. Kabsch. *A solution for the best rotation to relate two sets of vectors.* Acta Crystallographica 1976.

---

## License

MIT
