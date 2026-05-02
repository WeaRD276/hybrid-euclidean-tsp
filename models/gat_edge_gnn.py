from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEdgeGNN(nn.Module):
    """
    GAT-based edge classifier for TSP graphs.

    Input:
        data.x:          [num_nodes, 2]
        data.edge_index: [2, num_edges]
        data.edge_attr:  [num_edges, 3] = [dx, dy, dist]

    Output:
        logits: [num_edges]
    """

    def __init__(
        self,
        node_in_dim: int = 2,
        edge_in_dim: int = 3,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.node_encoder(x)

        h = self.gat1(h, edge_index)
        h = F.relu(h)

        h = self.gat2(h, edge_index)
        h = F.relu(h)

        src = edge_index[0]
        dst = edge_index[1]

        h_src = h[src]
        h_dst = h[dst]

        edge_features = torch.cat(
            [h_src, h_dst, edge_attr],
            dim=-1,
        )

        logits = self.edge_mlp(edge_features).squeeze(-1)
        return logits
