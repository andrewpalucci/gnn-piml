# gnn/train/train_gcn.py
import argparse, os, torch
from torch_geometric.data import Data
from gnn.models.gcn import GCNRegressor

def to_device(g: Data, device):
    g = g.clone()
    g.x = g.x.to(device)
    g.y = g.y.to(device)
    g.edge_index = g.edge_index.to(device)
    g.inc = g.inc.to(device)
    return g

def run_epoch(model, data_list, optimizer=None, pi_weight=0.0):
    mse = torch.nn.MSELoss()
    total = 0.0
    train = optimizer is not None
    if train:
        model.train()
    else:
        model.eval()
    with torch.set_grad_enabled(train):
        for g in data_list:
            pred = model(g.x, g.edge_index)
            loss = mse(pred, g.y)
            if pi_weight > 0.0:
                prior = g.x[:, 9:10]  # hess_prior feature
                loss = loss + pi_weight * mse(pred, prior)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.item())
    return total / max(1, len(data_list))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--val", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--pi", type=float, default=0.0, help="physics-informed weight")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="runs/model.pt")
    args = ap.parse_args()

    device = torch.device(args.device)
    train_set = [to_device(g, device) for g in torch.load(args.train)]
    val_set   = [to_device(g, device) for g in torch.load(args.val)]

    in_dim = train_set[0].x.shape[1]
    model = GCNRegressor(in_dim=in_dim, hidden=args.hidden, layers=args.layers).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best = float("inf")
    best_ckpt = None

    for ep in range(1, args.epochs+1):
        tr = run_epoch(model, train_set, optimizer=optim, pi_weight=args.pi)
        va = run_epoch(model, val_set, optimizer=None,    pi_weight=args.pi)
        print(f"Epoch {ep:03d} | train_loss={tr:.6f}  val_loss={va:.6f}")
        if va < best:
            best = va
            best_ckpt = {
                "model": model.state_dict(),
                "in_dim": in_dim,
                "hidden": args.hidden,
                "layers": args.layers,
                "pi": args.pi,
            }
            torch.save(best_ckpt, args.out)

    print(f"Saved best checkpoint -> {args.out} (val_loss={best:.6f})")

if __name__ == "__main__":
    main()
