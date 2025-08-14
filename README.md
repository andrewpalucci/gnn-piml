# FEM-GNN Error Estimation (Replication-Oriented)

Replicates a mesh→graph GNN that predicts per-element error for Laplace problems
(element = node, 3-NN edges; embeddings for position/area/BC). Compares a baseline
to a physics-informed variant. Time-bounded for poster-quality results.

## Layout
- `fem/meshing`  – Gmsh domain/mesh generation
- `fem/solves`   – FEniCSx Laplace solves (fine vs coarse) to compute elementwise targets
- `gnn/dataset`  – mesh→graph conversion (element nodes, 3-NN edges, edge distances)
- `gnn/models`   – GNNs (GCN baseline; PI-regularized variant)
- `gnn/train`    – training scripts / checkpoints
- `gnn/eval`     – validation & figure exporters
- `data/`        – raw meshes and serialized graphs
- `runs/`        – training outputs
- `figs/`        – poster-ready images

## Environment (Stage 1)
- Python 3.10 (conda)
- fenics-dolfinx, gmsh, meshio (conda-forge)
- torch, torch-geometric (CPU wheels)
- numpy, scipy, matplotlib, pyvista
