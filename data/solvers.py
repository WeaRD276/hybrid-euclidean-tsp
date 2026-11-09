"""Reference TSP solvers used to produce training labels and optimality gaps."""
from __future__ import annotations

import numpy as np

from data.held_karp import held_karp

# LKH works on integer distances, so coordinates are scaled before rounding.
LKH_SCALE = 1_000_000


def distance_matrix(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)


def tour_length(dist: np.ndarray, tour: np.ndarray) -> float:
    tour = np.asarray(tour)
    return float(dist[tour, np.roll(tour, -1)].sum())


def solve_held_karp(points: np.ndarray) -> np.ndarray:
    """Exact solver, O(n^2 2^n). Only practical for n <= ~13."""
    _, path, _, _, _ = held_karp([tuple(p) for p in np.asarray(points)])
    return np.asarray(path, dtype=np.int64)


def solve_lkh(points: np.ndarray, runs: int = 10) -> np.ndarray:
    """Near-optimal tour from LKH-3 (via elkai). Optimal in practice for n <= 100."""
    import elkai

    dist = distance_matrix(points)
    int_dist = np.rint(dist * LKH_SCALE).astype(np.int64).tolist()
    tour = elkai.DistanceMatrix(int_dist).solve_tsp(runs=runs)
    return np.asarray(tour[:-1], dtype=np.int64)


SOLVERS = {
    "held_karp": solve_held_karp,
    "lkh": solve_lkh,
}
