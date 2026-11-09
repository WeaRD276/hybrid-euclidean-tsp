"""
Edge "heatmaps" (importance in [0, 1] for every node pair) and the way they are
fused with distance into a greedy score (lower = better).
"""
from __future__ import annotations

import numpy as np


def prob_matrix(n: int, edge_index: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Dense symmetric heatmap from per-directed-edge probabilities.

    p(u, v) and p(v, u) are averaged; pairs missing from the graph get 0.
    """
    P = np.zeros((n, n))
    P[edge_index[0], edge_index[1]] = probs
    return (P + P.T) / 2.0


def knn_rank_heatmap(dist: np.ndarray) -> np.ndarray:
    """Non-learned heatmap: mean of 1/rank of j among i's neighbours and vice versa."""
    n = dist.shape[0]
    d = dist + np.diag(np.full(n, np.inf))
    rank = np.empty((n, n))
    rank[np.arange(n)[:, None], np.argsort(d, axis=1)] = np.arange(1, n + 1)
    H = 1.0 / rank
    np.fill_diagonal(H, 0.0)
    return (H + H.T) / 2.0


def softdist_heatmap(dist: np.ndarray, temperature: float | None = None) -> np.ndarray:
    """SoftDist (Xia et al., ICML 2024): row softmax of -d / tau, symmetrised."""
    n = dist.shape[0]
    d = dist + np.diag(np.full(n, np.inf))
    if temperature is None:
        temperature = float(d.min(axis=1).mean())
    logits = -d / temperature
    logits -= logits.max(axis=1, keepdims=True)
    H = np.exp(logits)
    H /= H.sum(axis=1, keepdims=True)
    return (H + H.T) / 2.0


def mean_nn_distance(dist: np.ndarray) -> float:
    n = dist.shape[0]
    return float((dist + np.diag(np.full(n, np.inf))).min(axis=1).mean())


def fused_score(dist: np.ndarray, heat: np.ndarray, lam: float, mode: str = "norm") -> np.ndarray:
    """
    raw:  d - lam * p                 (coursework formula; best lam depends on n)
    norm: d / mean_nn_dist - lam * p  (distance in units of the typical NN gap)
    """
    if mode == "raw":
        return dist - lam * heat
    if mode == "norm":
        return dist / mean_nn_distance(dist) - lam * heat
    raise ValueError(f"unknown score mode: {mode}")
