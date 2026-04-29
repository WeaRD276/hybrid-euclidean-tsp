from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

from data.dataset_generator import build_dataset
from heuristics.greedy_tour import greedy_tour, neural_guided_scores
from models.edge_gnn import EdgeGNN
from training.torch_dataset import to_pyg
from training.train_edge_classifier import (
    N,
    VAL_GRAPHS,
    VAL_SEED_OFFSET,
    CHECKPOINT_PATH,
)


LAMBDA_VALUES = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0]

from config import OUTPUTS_DIR
CSV_PATH = OUTPUTS_DIR / "tours" / "lambda_sweep_n10.csv"
PLOT_PATH = OUTPUTS_DIR / "tours" / "lambda_sweep_n10.png"


@torch.no_grad()
def predict_edge_probs(model: EdgeGNN, graph, device: torch.device) -> np.ndarray:
    model.eval()
    data = to_pyg(graph).to(device)
    logits = model(data)
    return torch.sigmoid(logits).cpu().numpy()


def optimal_tour_length(graph) -> float:
    mask = graph.y.astype(bool)
    return float(graph.edge_attr[:, 2][mask].sum() / 2.0)


def gap(method_length: float, opt_length: float) -> float:
    return (method_length - opt_length) / opt_length


def save_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "average_gap"])
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "lambda": f"{row['lambda']:.6f}",
                    "average_gap": f"{row['average_gap']:.6f}",
                }
            )


def save_plot(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lambdas = [row["lambda"] for row in rows]
    gaps = [row["average_gap"] for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lambdas, gaps, marker="o")

    ax.set_title("Validation average tour gap vs lambda")
    ax.set_xlabel("lambda")
    ax.set_ylabel("average gap")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run training first:\n"
            "python training/train_edge_classifier.py"
        )

    print("Building validation graphs...")
    val_graphs = build_dataset(
        num_samples=VAL_GRAPHS,
        n=N,
        seed_offset=VAL_SEED_OFFSET,
    )

    print("Loading model...")
    model = EdgeGNN(hidden_dim=64).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    probs_by_graph = [
        predict_edge_probs(model=model, graph=graph, device=device)
        for graph in val_graphs
    ]

    rows = []

    for lambda_ in LAMBDA_VALUES:
        gaps = []

        for graph, probs in zip(val_graphs, probs_by_graph):
            opt_len = optimal_tour_length(graph)

            scores = neural_guided_scores(
                graph=graph,
                edge_probs=probs,
                lambda_=lambda_,
            )

            _, tour_len = greedy_tour(
                graph=graph,
                edge_scores=scores,
            )

            gaps.append(gap(tour_len, opt_len))

        avg_gap = float(np.mean(gaps))
        rows.append(
            {
                "lambda": lambda_,
                "average_gap": avg_gap,
            }
        )

    best_row = min(rows, key=lambda row: row["average_gap"])

    save_csv(CSV_PATH, rows)
    save_plot(PLOT_PATH, rows)

    print()
    print("| lambda | average gap |")
    print("| ---: | ---: |")
    for row in rows:
        print(f"| {row['lambda']:.6f} | {row['average_gap']:.6f} |")

    print()
    print(
        f"Best validation lambda: {best_row['lambda']:.6f} "
        f"with average gap {best_row['average_gap']:.6f}"
    )
    print(f"Saved CSV to: {CSV_PATH}")
    print(f"Saved plot to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
