from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

# Allow running as:
# python experiments/evaluate_baselines.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_generator import build_dataset
from baselines.all_zeros import predict_all_zeros
from baselines.shortest_edges import predict_shortest_edges_unconstrained


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_metrics(conf: dict[str, int]) -> dict[str, float]:
    tp = conf["tp"]
    tn = conf["tn"]
    fp = conf["fp"]
    fn = conf["fn"]

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_baseline(
    dataset,
    predictor: Callable,
    name: str,
) -> None:
    total_conf = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for graph in dataset:
        y_true = graph.y.astype(np.int32)
        y_pred = predictor(graph).astype(np.int32)

        assert y_true.shape == y_pred.shape

        conf = compute_confusion(y_true, y_pred)
        for key in total_conf:
            total_conf[key] += conf[key]

    metrics = compute_metrics(total_conf)

    print(f"=== {name} ===")
    print(
        f"TP={total_conf['tp']}  TN={total_conf['tn']}  "
        f"FP={total_conf['fp']}  FN={total_conf['fn']}"
    )
    print(f"accuracy : {metrics['accuracy']:.6f}")
    print(f"precision: {metrics['precision']:.6f}")
    print(f"recall   : {metrics['recall']:.6f}")
    print(f"f1       : {metrics['f1']:.6f}")
    print()


def main() -> None:
    num_samples = 200
    n = 10
    seed_offset = 20000

    dataset = build_dataset(
        num_samples=num_samples,
        n=n,
        seed_offset=seed_offset,
    )

    positives = sum(int(np.sum(graph.y)) for graph in dataset)
    total_edges = sum(int(graph.y.shape[0]) for graph in dataset)

    print(f"Dataset size: {len(dataset)}")
    print(f"Nodes per graph: {n}")
    print(f"Total directed edges: {total_edges}")
    print(f"Total positive directed edges: {positives}")
    print(f"Positive rate: {positives / total_edges:.6f}")
    print()

    evaluate_baseline(dataset, predict_all_zeros, "All zeros")
    evaluate_baseline(dataset, predict_shortest_edges_unconstrained, "n shortest undirected edges")


if __name__ == "__main__":
    main()
