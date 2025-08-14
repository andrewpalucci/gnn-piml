# fem/solves/interp_error.py
"""
Compute per-element interpolation error for a harmonic analytic solution u(x,y)
on a unit-square mesh, and export element-level features + targets.

Saves: data/raw/unit_h{H}.npz with keys:
  incenters  (Ne, 2)
  areas      (Ne,)
  dleft, dright, dbot, dtop (Ne,)
  boundary_flag, boundary_val (Ne,)
  hess_prior (Ne,)        ~ h^2 * ||Hess(u)||_F at incenter
  target     (Ne,)        L2 interpolation error per element

Usage:
  python fem/solves/interp_error.py --h 0.07 --out data/raw/unit_h0.07.npz
"""
from mpi4py import MPI
from dolfinx import mesh as dmesh
from dolfinx import fem
from dolfinx.io import gmshio
import gmsh
import ufl
import numpy as np
import argparse
import math
from fem.meshing.make_mesh import build_unit_square as build_unit_square_mesh

# --- Analytic harmonic solution (Laplace) ---
# u(x,y) = sin(pi x)*sinh(pi y)/sinh(pi)
pi = math.pi
def u_exact_xy(x, y):
    return np.sin(pi*x) * np.sinh(pi*y) / np.sinh(pi)

def uxx_xy(x, y):
    return -(pi**2) * np.sin(pi*x) * np.sinh(pi*y) / np.sinh(pi)

def uyy_xy(x, y):
    return +(pi**2) * np.sin(pi*x) * np.sinh(pi*y) / np.sinh(pi)

def uxy_xy(x, y):
    return (pi**2) * np.cos(pi*x) * np.cosh(pi*y) / np.sinh(pi)

def hess_fro_xy(x, y):
    H11 = uxx_xy(x, y)
    H22 = uyy_xy(x, y)
    H12 = uxy_xy(x, y)
    return np.sqrt(H11**2 + H22**2 + 2*(H12**2))

# --- small geom helpers on triangles ---
def triangle_area(P):
    # P may be (3,3) (xyz) or (3,2) (xy). Use only xy for 2-D area.
    Px = P[:, :2]
    A, B, C = Px
    return 0.5 * abs((B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0]))

def triangle_incenter(P):
    # Use xy only; distances are Euclidean in 2-D
    Px = P[:, :2]
    A, B, C = Px
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)
    p = a + b + c
    return (a*A + b*B + c*C)/p  # returns (2,)

def quadrature_points_deg2(P):
    # Degree-2 barycentric rule in 2-D (xy only)
    Px = P[:, :2]
    A, B, C = Px
    w = np.array([1/3, 1/3, 1/3], dtype=float)
    q1 = (2/3)*A + (1/6)*B + (1/6)*C
    q2 = (1/6)*A + (2/3)*B + (1/6)*C
    q3 = (1/6)*A + (1/6)*B + (2/3)*C
    return np.stack([q1, q2, q3], axis=0), w

def per_element_interpolation_error(msh: dmesh.Mesh):
    """
    Compute L2 interpolation error of u_exact over each triangle using P1 nodal interpolation,
    evaluated via barycentric weights (no Function.eval calls).
    """
    # Use xy only for geometry math
    X = msh.geometry.x[:, :2]
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)

    Ne = msh.topology.index_map(tdim).size_local
    areas   = np.zeros(Ne, dtype=float)
    centers = np.zeros((Ne, 2), dtype=float)
    target  = np.zeros(Ne, dtype=float)

    dleft = np.zeros(Ne); dright = np.zeros(Ne)
    dbot  = np.zeros(Ne); dtop   = np.zeros(Ne)
    bflag = np.zeros(Ne); bval   = np.zeros(Ne)
    hprior= np.zeros(Ne)

    def cross2(a, b):
        return a[0]*b[1] - a[1]*b[0]

    def uh_at_qpoints(P, u_vtx, Q):
        """P:(3,2) triangle vertices, u_vtx:(3,) vertex values, Q:(m,2) points -> (m,)"""
        A, B, C = P
        denom = cross2(B - A, C - A)  # oriented 2*area
        out = np.empty(Q.shape[0], dtype=float)
        for i, q in enumerate(Q):
            l1 = cross2(B - q, C - q) / denom
            l2 = cross2(C - q, A - q) / denom
            l3 = 1.0 - l1 - l2
            out[i] = l1*u_vtx[0] + l2*u_vtx[1] + l3*u_vtx[2]
        return out

    for c in range(Ne):
        vs = c2v.links(c)
        P  = X[vs]  # (3,2)
        A  = triangle_area(P)
        C  = triangle_incenter(P)
        areas[c]   = A
        centers[c] = C

        x, y = C
        dleft[c], dright[c] = x, 1.0 - x
        dbot[c],  dtop[c]   = y, 1.0 - y
        on_bdry = np.any(np.isclose(P[:,0], 0.0) | np.isclose(P[:,0], 1.0) |
                         np.isclose(P[:,1], 0.0) | np.isclose(P[:,1], 1.0))
        bflag[c] = 1.0 if on_bdry else 0.0
        bval[c]  = u_exact_xy(x, y) if on_bdry else 0.0

        h = math.sqrt(A)
        hprior[c] = (h**2) * hess_fro_xy(x, y)

        # Quadrature (degree-2) and interpolation error
        Q, w = quadrature_points_deg2(P)               # (3,2), (3,)
        u_vtx = u_exact_xy(P[:,0], P[:,1])            # exact at vertices
        uhq = uh_at_qpoints(P, u_vtx, Q)               # interpolant at quad points
        ue3 = u_exact_xy(Q[:,0], Q[:,1])               # exact at quad points

        err_sq = A * (1.0/3.0) * np.sum((ue3 - uhq)**2)
        target[c] = math.sqrt(max(err_sq, 0.0))

    return centers, areas, dleft, dright, dbot, dtop, bflag, bval, hprior, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.07, help="Target element size (smaller=higher res)")
    ap.add_argument("--out", type=str, default="data/raw/unit_h0.07.npz")
    args = ap.parse_args()

    msh, _facet_tags = build_unit_square_mesh(args.h)
    if msh.comm.rank != 0:
        return
    (centers, areas, dL, dR, dB, dT, bflag, bval, hprior, target) = per_element_interpolation_error(msh)

    # Save compact .npz for Stage 4
    import os
    os.makedirs("data/raw", exist_ok=True)
    np.savez_compressed(
        args.out,
        incenters=centers,
        areas=areas,
        dleft=dL, dright=dR, dbot=dB, dtop=dT,
        boundary_flag=bflag, boundary_val=bval,
        hess_prior=hprior, target=target
    )
    print(f"✅ Saved {args.out} with {len(target)} elements")

if __name__ == "__main__":
    main()

