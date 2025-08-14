# gnn/eval/visualize.py
import argparse, os, torch, numpy as np, matplotlib.pyplot as plt
from gnn.models.gcn import GCNRegressor

def make_fig(inc, truth, pred, out):
    plt.figure(figsize=(8,3.5))
    plt.subplot(1,2,1)
    sc1 = plt.scatter(inc[:,0], inc[:,1], c=truth, s=8)
    plt.title("Ground truth error"); plt.colorbar(sc1, fraction=0.046)
    plt.subplot(1,2,2)
    sc2 = plt.scatter(inc[:,0], inc[:,1], c=pred, s=8)
    plt.title("Predicted error"); plt.colorbar(sc2, fraction=0.046)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"Saved {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out", type=str, default="figs/sample.png")
    args = ap.parse_args()

    graphs = torch.load(args.dataset)
    data = graphs[args.idx]
    ck = torch.load(args.checkpoint, map_location="cpu")
    model = GCNRegressor(in_dim=ck["in_dim"], hidden=ck["hidden"], layers=ck["layers"])
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).cpu().numpy().ravel()
    truth = data.y.cpu().numpy().ravel()
    inc   = data.inc.cpu().numpy()
    make_fig(inc, truth, pred, args.out)

if __name__ == "__main__":
    main()
