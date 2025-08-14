# gnn/models/gcn.py
import torch
from torch import nn
from torch_geometric.nn import GCNConv, BatchNorm

class GCNRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 1, layers: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        self.bns.append(BatchNorm(hidden))
        for _ in range(layers-1):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(BatchNorm(hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x, edge_index):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = torch.relu(h)
        return self.head(h)
