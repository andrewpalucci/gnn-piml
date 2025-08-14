# scripts/make_raw_dataset.py
import argparse, os, numpy as np
from fem.meshing.make_mesh import build_unit_square
from fem.solves.interp_error import per_element_interpolation_error

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of samples")
    ap.add_argument("--hmin", type=float, default=0.05)
    ap.add_argument("--hmax", type=float, default=0.12)
    ap.add_argument("--split", type=str, default="train", choices=["train","val","test"])
    ap.add_argument("--outdir", type=str, default="data/raw")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    for i in range(args.n):
        h = float(rng.uniform(args.hmin, args.hmax))
        msh, _facet = build_unit_square(h)
        # rank 0 only (single-proc run anyway)
        if msh.comm.rank != 0:
            continue
        (centers, areas, dL, dR, dB, dT, bflag, bval, hprior, target) = per_element_interpolation_error(msh)
        path = os.path.join(args.outdir, f"unit_{args.split}_{i:04d}_h{h:.4f}.npz")
        np.savez_compressed(
            path,
            incenters=centers, areas=areas,
            dleft=dL, dright=dR, dbot=dB, dtop=dT,
            boundary_flag=bflag, boundary_val=bval,
            hess_prior=hprior, target=target
        )
        print(f"[{i+1}/{args.n}] saved {path}  (elements={len(target)})")

if __name__ == "__main__":
    main()
