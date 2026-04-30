from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.serialization import load_split_samples
from heuristics.greedy_tour import (
    greedy_tour,
    distance_scores,
    neural_guided_scores,
)
from models.edge_gnn import EdgeGNN
from training.torch_dataset import to_pyg
from training.train_edge_classifier import (
    N,
    CHECKPOINT_PATH,
)
from config import OUTPUTS_DIR, SAVED_DATA_DIR


LAMBDA = 5.0

OUT_PATH = OUTPUTS_DIR / "tours" / "tour_results_n10.md"
TEST_DIR = SAVED_DATA_DIR / f"n_{N}" / "test"


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


def save_results(path: str | Path, rows: list[tuple[str, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Tour Construction Results for n = 10")
    lines.append("")
    lines.append(f"- Selected lambda: {LAMBDA}")
    lines.append("")
    lines.append("| method | average gap |")
    lines.append("| --- | ---: |")

    for method, avg_gap in rows:
        lines.append(f"| {method} | {avg_gap:.6f} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run training first:\n"
            "python training/train_edge_classifier.py"
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Saved test split not found: {TEST_DIR}\n"
            "Generate dataset first:\n"
            "python data/dataset_generator.py"
        )

    print("Loading test graphs from files...")
    test_graphs = load_split_samples(TEST_DIR)

    print("Loading model...")
    model = EdgeGNN(hidden_dim=64).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    distance_gaps = []
    neural_gaps = []

    for graph in test_graphs:
        opt_len = optimal_tour_length(graph)

        _, distance_len = greedy_tour(
            graph=graph,
            edge_scores=distance_scores(graph),
        )
        distance_gaps.append(gap(distance_len, opt_len))

        probs = predict_edge_probs(
            model=model,
            graph=graph,
            device=device,
        )

        neural_scores = neural_guided_scores(
            graph=graph,
            edge_probs=probs,
            lambda_=LAMBDA,
        )

        _, neural_len = greedy_tour(
            graph=graph,
            edge_scores=neural_scores,
        )
        neural_gaps.append(gap(neural_len, opt_len))

    rows = [
        ("greedy distance only", float(np.mean(distance_gaps))),
        (f"neural greedy lambda={LAMBDA}", float(np.mean(neural_gaps))),
    ]

    print()
    print("| method | average gap |")
    print("| --- | ---: |")
    for method, avg_gap in rows:
        print(f"| {method} | {avg_gap:.6f} |")

    save_results(OUT_PATH, rows)

    print()
    print(f"Saved results to: {OUT_PATH}")


if __name__ == "__main__":
    main()
