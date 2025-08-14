# gnn/dataset/mesh_to_graph.py
import argparse, glob, os, numpy as np, torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree

def npz_to_graph(path: str, k: int = 3) -> Data:
    Z = np.load(path)
    inc = Z["incenters"]                  # (Ne, 2)
    areas = Z["areas"]                    # (Ne,)
    dL, dR, dB, dT = Z["dleft"], Z["dright"], Z["dbot"], Z["dtop"]
    bflag, bval = Z["boundary_flag"], Z["boundary_val"]
    hprior = Z["hess_prior"]
    target = Z["target"]

    # Node feature matrix (mirror the paper’s spirit)
    X = np.column_stack([
        inc[:,0], inc[:,1],               # incenter (x,y)
        dL, dR, dB, dT,                   # distances to 4 sides
        areas,                            # area
        bflag, bval,                      # boundary flag/value
        hprior                            # physics prior
    ]).astype(np.float32)

    # Build 3-NN edges (directed)
    tree = cKDTree(inc)
    dists, idxs = tree.query(inc, k=k+1)  # self + k neighbors
    src, dst = [], []
    for i in range(inc.shape[0]):
        for j in range(1, k+1):
            jj = idxs[i, j]
            if jj == i:  # should not happen after skipping 0
                continue
            src.append(i); dst.append(jj)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(
        x=torch.from_numpy(X),
        edge_index=edge_index,
        y=torch.from_numpy(target.astype(np.float32)).view(-1,1)
    )
    # stash incenters for visualization
    data.inc = torch.from_numpy(inc.astype(np.float32))
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, required=True, help='e.g. "data/raw/unit_train_*.npz"')
    ap.add_argument("--out", type=str, required=True, help="output .pt path")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    graphs = [npz_to_graph(p, k=args.k) for p in paths]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(graphs, args.out)
    print(f"Saved {len(graphs)} graphs -> {args.out}")

if __name__ == "__main__":
    main()
