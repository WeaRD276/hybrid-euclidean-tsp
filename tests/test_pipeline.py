import numpy as np
import pytest

from data.graphs import build_graph, knn_mask, tour_adjacency
from data.solvers import distance_matrix, solve_held_karp, solve_lkh, tour_length
from heuristics.classical import farthest_insertion, greedy_edge, nearest_neighbour, two_opt
from heuristics.greedy_tour import greedy_tour
from heuristics.scores import fused_score, knn_rank_heatmap, prob_matrix
from data.graph_builder import TSPGraphData


def instance(n, seed=0):
    points = np.random.default_rng(seed).random((n, 2))
    return points, distance_matrix(points)


def is_tour(tour, n):
    return sorted(tour.tolist()) == list(range(n))


@pytest.mark.parametrize("seed", range(20))
def test_lkh_matches_held_karp(seed):
    points, dist = instance(10, seed)
    assert tour_length(dist, solve_lkh(points)) == pytest.approx(tour_length(dist, solve_held_karp(points)))


@pytest.mark.parametrize("heuristic", [greedy_edge, nearest_neighbour, farthest_insertion])
def test_heuristics_return_valid_tours(heuristic):
    for seed in range(10):
        _, dist = instance(25, seed)
        assert is_tour(heuristic(dist), 25)


def test_two_opt_never_worse():
    for seed in range(10):
        _, dist = instance(30, seed)
        start = nearest_neighbour(dist)
        improved = two_opt(dist, start)
        assert is_tour(improved, 30)
        assert tour_length(dist, improved) <= tour_length(dist, start) + 1e-12


def test_greedy_matches_coursework_implementation():
    points, dist = instance(9, 3)
    graph = TSPGraphData.from_points(points.tolist())
    _, old_length = greedy_tour(graph, graph.edge_attr[:, 2])
    assert tour_length(dist, greedy_edge(dist)) == pytest.approx(old_length)


def test_lambda_zero_is_plain_greedy():
    _, dist = instance(20, 1)
    heat = knn_rank_heatmap(dist)
    for mode in ("raw", "norm"):
        hybrid = greedy_edge(dist, fused_score(dist, heat, 0.0, mode))
        assert tour_length(dist, hybrid) == pytest.approx(tour_length(dist, greedy_edge(dist)))


def test_perfect_heatmap_recovers_reference_tour():
    points, dist = instance(15, 2)
    tour = solve_lkh(points)
    heat = tour_adjacency(tour).astype(float)
    hybrid = greedy_edge(dist, fused_score(dist, heat, 1e6, "raw"))
    assert tour_length(dist, hybrid) == pytest.approx(tour_length(dist, tour))


def test_graph_labels_and_prob_matrix_symmetry():
    points, dist = instance(12, 4)
    tour = solve_lkh(points)
    for graph, k in (("complete", None), ("knn", 4)):
        data = build_graph(points, tour, graph, k)
        ei = data.edge_index.numpy()
        assert data.y.sum().item() <= 2 * 12
        P = prob_matrix(12, ei, np.random.default_rng(0).random(ei.shape[1]))
        assert np.allclose(P, P.T)
    assert knn_mask(dist, 4).sum(axis=1).min() >= 4
