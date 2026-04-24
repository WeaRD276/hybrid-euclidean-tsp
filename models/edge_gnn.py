from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class EdgeGNN(nn.Module):
    """
    Minimal GNN for edge classification in TSP graphs.

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
    ):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.conv1 = GraphConv(hidden_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

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

        h = self.conv1(h, edge_index)
        h = F.relu(h)

        h = self.conv2(h, edge_index)
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


def compute_pos_weight(num_nodes: int) -> torch.Tensor:
    """
    For one complete directed graph:
        total edges = n * (n - 1)
        positive edges = 2 * n
        negative edges = total - positive

    pos_weight = negative / positive
    """
    total_edges = num_nodes * (num_nodes - 1)
    positive_edges = 2 * num_nodes
    negative_edges = total_edges - positive_edges

    return torch.tensor(negative_edges / positive_edges, dtype=torch.float32)


def make_loss_fn(num_nodes: int, device: torch.device | str = "cpu") -> nn.Module:
    pos_weight = compute_pos_weight(num_nodes).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
