"""
Classical construction and improvement heuristics on a distance matrix.

All functions return a tour as a node order (np.ndarray of length n).
"""
from __future__ import annotations

import numpy as np


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, v: int) -> int:
        while self.parent[v] != v:
            self.parent[v] = self.parent[self.parent[v]]
            v = self.parent[v]
        return v

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def edges_to_tour(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    neighbours = [[] for _ in range(n)]
    for u, v in edges:
        neighbours[u].append(v)
        neighbours[v].append(u)

    tour = [0]
    prev, cur = -1, 0
    for _ in range(n - 1):
        a, b = neighbours[cur]
        prev, cur = cur, (b if a == prev else a)
        tour.append(cur)
    return np.asarray(tour, dtype=np.int64)


def greedy_edge(dist: np.ndarray, score: np.ndarray | None = None) -> np.ndarray:
    """
    Greedy edge matching: take undirected edges in increasing score order and keep
    an edge if both endpoints have degree < 2 and it closes no premature cycle.
    Ties in score are broken by distance, so score = dist gives plain greedy.
    """
    n = dist.shape[0]
    if score is None:
        score = dist

    iu, ju = np.triu_indices(n, k=1)
    order = np.lexsort((dist[iu, ju], score[iu, ju]))

    degree = np.zeros(n, dtype=np.int64)
    dsu = DSU(n)
    edges: list[tuple[int, int]] = []

    for idx in order:
        u, v = int(iu[idx]), int(ju[idx])
        if degree[u] >= 2 or degree[v] >= 2:
            continue
        closes_cycle = dsu.find(u) == dsu.find(v)
        if closes_cycle and len(edges) != n - 1:
            continue

        edges.append((u, v))
        degree[u] += 1
        degree[v] += 1
        if not closes_cycle:
            dsu.union(u, v)
        if len(edges) == n:
            break

    return edges_to_tour(n, edges)


def nearest_neighbour(dist: np.ndarray, start: int = 0) -> np.ndarray:
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        d = np.where(visited, np.inf, dist[tour[-1]])
        nxt = int(np.argmin(d))
        tour.append(nxt)
        visited[nxt] = True
    return np.asarray(tour, dtype=np.int64)


def farthest_insertion(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    tour = [int(i), int(j)]
    in_tour = np.zeros(n, dtype=bool)
    in_tour[tour] = True
    # distance from every node to its nearest tour node
    closest = np.minimum(dist[i], dist[j])

    for _ in range(n - 2):
        candidate = int(np.argmax(np.where(in_tour, -np.inf, closest)))
        t = np.asarray(tour)
        nxt = np.roll(t, -1)
        cost = dist[t, candidate] + dist[candidate, nxt] - dist[t, nxt]
        pos = int(np.argmin(cost))
        tour.insert(pos + 1, candidate)
        in_tour[candidate] = True
        closest = np.minimum(closest, dist[candidate])

    return np.asarray(tour, dtype=np.int64)


def two_opt(dist: np.ndarray, tour: np.ndarray, max_iters: int = 10_000) -> np.ndarray:
    """Best-improvement 2-opt until no improving move exists."""
    tour = np.asarray(tour).copy()
    n = len(tour)
    for _ in range(max_iters):
        a = tour
        b = np.roll(tour, -1)
        # delta[i, j]: replace edges (a_i, b_i), (a_j, b_j) by (a_i, a_j), (b_i, b_j)
        delta = (
            dist[a[:, None], a[None, :]]
            + dist[b[:, None], b[None, :]]
            - dist[a, b][:, None]
            - dist[a, b][None, :]
        )
        delta = np.triu(delta, k=2)
        delta[0, n - 1] = 0.0
        i, j = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[i, j] >= -1e-10:
            break
        tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
    return tour
