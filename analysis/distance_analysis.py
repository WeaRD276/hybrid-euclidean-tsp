from __future__ import annotations

from pathlib import Path

import numpy as np


def mean_distance_for_mask(graph, mask: np.ndarray) -> float:
    distances = graph.edge_attr[:, 2]
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return 0.0

    return float(distances[mask].mean())


def compute_distance_stats(
    graph,
    baseline_pred: np.ndarray,
    model_pred: np.ndarray,
) -> dict[str, float]:
    all_mask = np.ones_like(graph.y, dtype=bool)
    optimal_mask = graph.y.astype(bool)

    return {
        "all_edges": mean_distance_for_mask(graph, all_mask),
        "optimal_edges": mean_distance_for_mask(graph, optimal_mask),
        "baseline_predicted_edges": mean_distance_for_mask(graph, baseline_pred),
        "model_predicted_edges": mean_distance_for_mask(graph, model_pred),
    }


def average_distance_stats(
    graphs,
    baseline_predictions,
    model_predictions,
) -> dict[str, float]:
    totals = {
        "all_edges": 0.0,
        "optimal_edges": 0.0,
        "baseline_predicted_edges": 0.0,
        "model_predicted_edges": 0.0,
    }

    n = len(graphs)

    for graph, baseline_pred, model_pred in zip(
        graphs,
        baseline_predictions,
        model_predictions,
    ):
        stats = compute_distance_stats(
            graph=graph,
            baseline_pred=baseline_pred,
            model_pred=model_pred,
        )

        for key in totals:
            totals[key] += stats[key]

    return {key: value / n for key, value in totals.items()}


def save_distance_analysis(
    path: str | Path,
    stats: dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Distance Analysis")
    lines.append("")
    lines.append("| edge group | average distance |")
    lines.append("| --- | ---: |")
    lines.append(f"| all edges | {stats['all_edges']:.6f} |")
    lines.append(f"| optimal edges | {stats['optimal_edges']:.6f} |")
    lines.append(
        f"| baseline predicted edges | "
        f"{stats['baseline_predicted_edges']:.6f} |"
    )
    lines.append(
        f"| model predicted edges | "
        f"{stats['model_predicted_edges']:.6f} |"
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_distance_analysis(
    graphs,
    baseline_predictions,
    model_predictions,
    out_path: str | Path,
) -> dict[str, float]:
    stats = average_distance_stats(
        graphs=graphs,
        baseline_predictions=baseline_predictions,
        model_predictions=model_predictions,
    )

    save_distance_analysis(
        path=out_path,
        stats=stats,
    )

    return stats
