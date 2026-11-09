"""Edge-level and tour-level metrics, plus the statistics used in every comparison."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def _pairs(heat: np.ndarray, adj: np.ndarray):
    iu, ju = np.triu_indices(heat.shape[0], k=1)
    return heat[iu, ju], adj[iu, ju]


def edge_scores_and_labels(heats, adjs) -> tuple[np.ndarray, np.ndarray]:
    """Flatten dense heatmaps and tour adjacencies over undirected pairs i < j."""
    scores, labels = zip(*(_pairs(h, a) for h, a in zip(heats, adjs)))
    return np.concatenate(scores), np.concatenate(labels)


def prf(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def best_f1_threshold(labels, scores, thresholds=np.arange(0.05, 0.96, 0.05)) -> float:
    return float(max(thresholds, key=lambda t: prf(labels, scores, t)["f1"]))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    hits = labels[order].astype(float)
    precision_at = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision_at * hits).sum() / hits.sum())


def precision_at_n(heats, adjs) -> float:
    """Share of tour edges among the n highest-scoring pairs (n = number of tour edges)."""
    vals = []
    for h, a in zip(heats, adjs):
        s, y = _pairs(h, a)
        top = np.argsort(-s, kind="stable")[: h.shape[0]]
        vals.append(y[top].mean())
    return float(np.mean(vals))


def recall_at_k(heats, adjs, k: int) -> float:
    """Share of each node's two tour neighbours that lie among its top-k heatmap entries."""
    hit = total = 0
    for h, a in zip(heats, adjs):
        h = h - np.diag(np.full(h.shape[0], np.inf))
        top = np.argsort(-h, axis=1, kind="stable")[:, :k]
        hit += int(np.take_along_axis(a, top, axis=1).sum())
        total += int(a.sum())
    return hit / total


def top_n_length_ratio(heats, adjs, dists) -> float:
    """
    Mean length of the n highest-scoring pairs divided by the mean tour-edge length.
    Below 1 means the heatmap prefers edges shorter than the optimal ones.
    """
    ratios = []
    for h, a, d in zip(heats, adjs, dists):
        s, _ = _pairs(h, a)
        dd, y = _pairs(d, a)
        top = np.argsort(-s, kind="stable")[: h.shape[0]]
        ratios.append(dd[top].mean() / dd[y].mean())
    return float(np.mean(ratios))


def bootstrap_ci(values: np.ndarray, num_samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(num_samples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_comparison(gaps: np.ndarray, reference: np.ndarray, tol: float = 1e-9) -> dict[str, float]:
    diff = gaps - reference
    wins = int((diff < -tol).sum())
    losses = int((diff > tol).sum())
    ties = len(diff) - wins - losses
    p = float(wilcoxon(gaps, reference).pvalue) if wins + losses else 1.0
    return {"wins": wins, "ties": ties, "losses": losses, "wilcoxon_p": p}
