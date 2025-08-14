# fem/meshing/make_mesh.py
"""
Build a unit-square triangular mesh with facet tags:
1=left (x=0), 2=right (x=1), 3=bottom (y=0), 4=top (y=1)

Usage:
  python fem/meshing/make_mesh.py --h 0.07
"""
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx import mesh as dmesh
import gmsh
import argparse

def build_unit_square(h: float = 0.07):
    gmsh.initialize()
    gmsh.model.add("unit_square")
    # Geometry
    p1 = gmsh.model.geo.addPoint(0, 0, 0, h)
    p2 = gmsh.model.geo.addPoint(1, 0, 0, h)
    p3 = gmsh.model.geo.addPoint(1, 1, 0, h)
    p4 = gmsh.model.geo.addPoint(0, 1, 0, h)

    l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # right
    l3 = gmsh.model.geo.addLine(p3, p4)  # top
    l4 = gmsh.model.geo.addLine(p4, p1)  # left
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()

    # Physical tags for facets (so we can locate boundaries later)
    bottom_tag, right_tag, top_tag, left_tag = 3, 2, 4, 1
    gmsh.model.addPhysicalGroup(1, [l4], left_tag)
    gmsh.model.setPhysicalName(1, left_tag, "left")
    gmsh.model.addPhysicalGroup(1, [l2], right_tag)
    gmsh.model.setPhysicalName(1, right_tag, "right")
    gmsh.model.addPhysicalGroup(1, [l1], bottom_tag)
    gmsh.model.setPhysicalName(1, bottom_tag, "bottom")
    gmsh.model.addPhysicalGroup(1, [l3], top_tag)
    gmsh.model.setPhysicalName(1, top_tag, "top")

    # Tag the surface as well (not strictly needed for this project)
    gmsh.model.addPhysicalGroup(2, [s], 10)
    gmsh.model.setPhysicalName(2, 10, "domain")

    gmsh.model.mesh.generate(2)

    # Convert to DOLFINx mesh + meshtags
    msh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return msh, facet_tags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.07, help="Target element size (smaller=higher res)")
    args = ap.parse_args()

    msh, facet_tags = build_unit_square(args.h)
    if msh.comm.rank == 0:
        print(f"✅ Mesh OK: {msh.topology.index_map(msh.topology.dim).size_local} elements, h≈{args.h}")
        # Quick facet counts per tag (sanity)
        msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
        import numpy as np
        unique, counts = np.unique(facet_tags.values, return_counts=True)
        print("Facet tags (id:count):", dict(zip(map(int, unique), map(int, counts))))

if __name__ == "__main__":
    main()

