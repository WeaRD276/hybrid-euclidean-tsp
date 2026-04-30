from __future__ import annotations

import numpy as np


def distance_scores(graph) -> np.ndarray:
    return graph.edge_attr[:, 2].copy()

def neural_guided_scores(
    graph,
    edge_probs: np.ndarray,
    lambda_: float,
) -> np.ndarray:
    distances = graph.edge_attr[:, 2]

    if edge_probs.shape[0] != distances.shape[0]:
        raise ValueError(
            f"edge_probs has shape {edge_probs.shape}, "
            f"but graph has {distances.shape[0]} edges"
        )

    return distances - lambda_ * edge_probs

class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, v: int) -> int:
        while self.parent[v] != v:
            self.parent[v] = self.parent[self.parent[v]]
            v = self.parent[v]
        return v

    def union(self, a: int, b: int) -> bool:
        a = self.find(a)
        b = self.find(b)

        if a == b:
            return False

        if self.size[a] < self.size[b]:
            a, b = b, a

        self.parent[b] = a
        self.size[a] += self.size[b]
        return True


def _get_undirected_edges(graph, edge_scores: np.ndarray) -> list[tuple[float, float, int, int]]:
    """
    Returns undirected edges as:
        (score, distance, u, v)

    Maybe latet I will use only one direction u < v.
    """
    edge_index = graph.edge_index
    distances = graph.edge_attr[:, 2]

    edges = []

    for k in range(edge_index.shape[1]):
        u = int(edge_index[0, k])
        v = int(edge_index[1, k])

        if u < v:
            score = float(edge_scores[k])
            distance = float(distances[k])

            edges.append((score, distance, u, v))

    return edges


def greedy_tour(graph, edge_scores: np.ndarray):
    """
    Greedy TSP tour construction.

    Input:
        graph:
            TSPGraphData with:
                graph.x          [n, 2]
                graph.edge_index [2, E]
                graph.edge_attr  [E, 3], where edge_attr[:, 2] is distance

        edge_scores:
            score for every directed edge [E]
            lower score = better edge

            Examples:
                - distance score: graph.edge_attr[:, 2]
                - neural score: -predicted_probability

    Output:
        selected_edges:
            set of undirected edges (u, v), u < v

        tour_length:
            total Euclidean length of selected tour
    """
    n = graph.x.shape[0]

    if edge_scores.shape[0] != graph.edge_index.shape[1]:
        raise ValueError(
            f"edge_scores has shape {edge_scores.shape}, "
            f"but graph has {graph.edge_index.shape[1]} directed edges"
        )

    edges = _get_undirected_edges(graph, edge_scores)
    edges.sort(key=lambda item: item[0])

    degree = [0] * n
    dsu = DSU(n)

    selected_edges: set[tuple[int, int]] = set()
    tour_length = 0.0

    for score, distance, u, v in edges:
        if degree[u] >= 2 or degree[v] >= 2:
            continue

        creates_cycle = dsu.find(u) == dsu.find(v)

        # Before the last edge, cycles are forbidden.
        # The only allowed cycle is the final Hamiltonian cycle with n edges.
        if creates_cycle and len(selected_edges) != n - 1:
            continue

        selected_edges.add((u, v))
        tour_length += distance

        degree[u] += 1
        degree[v] += 1

        if not creates_cycle:
            dsu.union(u, v)

        if len(selected_edges) == n:
            break

    if len(selected_edges) != n:
        raise RuntimeError(
            f"Failed to construct Hamiltonian cycle: "
            f"selected {len(selected_edges)} edges instead of {n}"
        )

    if any(d != 2 for d in degree):
        raise RuntimeError(f"Invalid tour degrees: {degree}")

    return selected_edges, tour_length

