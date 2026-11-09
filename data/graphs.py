"""
PyG graphs built from raw instances (points + reference tour).

Two graph types:
    complete  every ordered pair (i, j), i != j  (the coursework setting)
    knn       (i, j) where j is among the k nearest neighbours of i or vice versa

Both store directed edges in both directions with
    x          [n, 2]   coordinates
    edge_attr  [E, 3]   dx, dy, dist
    y          [E]      1 if the undirected edge is in the reference tour
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from data.solvers import distance_matrix


def tour_adjacency(tour: np.ndarray) -> np.ndarray:
    n = len(tour)
    adj = np.zeros((n, n), dtype=bool)
    nxt = np.roll(tour, -1)
    adj[tour, nxt] = True
    adj[nxt, tour] = True
    return adj


def knn_mask(dist: np.ndarray, k: int) -> np.ndarray:
    n = dist.shape[0]
    k = min(k, n - 1)
    d = dist + np.diag(np.full(n, np.inf))
    nearest = np.argsort(d, axis=1)[:, :k]
    mask = np.zeros((n, n), dtype=bool)
    mask[np.repeat(np.arange(n), k), nearest.ravel()] = True
    return mask | mask.T


def edge_mask(dist: np.ndarray, graph: str, k: int | None) -> np.ndarray:
    n = dist.shape[0]
    if graph == "complete":
        return ~np.eye(n, dtype=bool)
    if graph == "knn":
        if k is None:
            raise ValueError("knn graph needs k")
        return knn_mask(dist, k)
    raise ValueError(f"unknown graph type: {graph}")


def build_graph(points: np.ndarray, tour: np.ndarray, graph: str = "complete", k: int | None = None) -> Data:
    dist = distance_matrix(points)
    src, dst = np.nonzero(edge_mask(dist, graph, k))
    delta = points[dst] - points[src]
    edge_attr = np.column_stack([delta, dist[src, dst]])
    y = tour_adjacency(tour)[src, dst]

    data = Data(
        x=torch.tensor(points, dtype=torch.float32),
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
    )
    data.num_nodes = len(points)
    return data


def build_graphs(split: dict[str, np.ndarray], graph: str = "complete", k: int | None = None) -> list[Data]:
    return [build_graph(p, t, graph, k) for p, t in zip(split["points"], split["tours"])]


def tour_edge_coverage(split: dict[str, np.ndarray], graph: str, k: int | None) -> float:
    """Share of reference-tour edges present in the graph (upper bound for kNN methods)."""
    covered = total = 0
    for points, tour in zip(split["points"], split["tours"]):
        adj = tour_adjacency(tour)
        mask = edge_mask(distance_matrix(points), graph, k)
        covered += int((adj & mask).sum())
        total += int(adj.sum())
    return covered / total
