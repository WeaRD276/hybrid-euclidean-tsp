from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_generator import build_dataset
from heuristics.greedy_tour import (
    greedy_tour,
    distance_scores,
    neural_guided_scores,
)
from models.edge_gnn import EdgeGNN
from training.torch_dataset import to_pyg
from training.train_edge_classifier import (
    N,
    TEST_GRAPHS,
    TEST_SEED_OFFSET,
    CHECKPOINT_PATH,
)

from config import OUTPUTS_DIR
OUT_PATH = OUTPUTS_DIR / "experiments" / "tour_results_n10.md"

LAMBDA_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]


@torch.no_grad()
def predict_edge_probs(model: EdgeGNN, graph, device: torch.device) -> np.ndarray:
    model.eval()

    data = to_pyg(graph).to(device)
    logits = model(data)
    probs = torch.sigmoid(logits)

    return probs.cpu().numpy()


def optimal_tour_length(graph) -> float:
    """
    Since graph.y marks both directions of each optimal edge,
    summing all positive directed distances counts each tour edge twice.
    """
    mask = graph.y.astype(bool)
    return float(graph.edge_attr[:, 2][mask].sum() / 2.0)


def gap(method_length: float, opt_length: float) -> float:
    return (method_length - opt_length) / opt_length


def save_results(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Tour Construction Results for n = 10")
    lines.append("")
    lines.append("| method | average gap |")
    lines.append("| --- | ---: |")

    for method, avg_gap in rows:
        lines.append(f"| {method} | {avg_gap:.6f} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    torch.manual_seed(47)
    np.random.seed(47)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run training first:\n"
            "python training/train_edge_classifier.py"
        )

    print("Building test graphs...")
    test_graphs = build_dataset(
        num_samples=TEST_GRAPHS,
        n=N,
        seed_offset=TEST_SEED_OFFSET,
    )

    print("Loading model...")
    model = EdgeGNN(hidden_dim=64).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    distance_gaps = []
    neural_gaps_by_lambda = {lambda_: [] for lambda_ in LAMBDA_VALUES}

    for graph in test_graphs:
        opt_len = optimal_tour_length(graph)

        # 1. Pure greedy by distance
        _, dist_len = greedy_tour(
            graph=graph,
            edge_scores=distance_scores(graph),
        )
        distance_gaps.append(gap(dist_len, opt_len))

        # 2. Neural-guided greedy
        probs = predict_edge_probs(
            model=model,
            graph=graph,
            device=device,
        )

        for lambda_ in LAMBDA_VALUES:
            scores = neural_guided_scores(
                graph=graph,
                edge_probs=probs,
                lambda_=lambda_,
            )

            _, neural_len = greedy_tour(
                graph=graph,
                edge_scores=scores,
            )

            neural_gaps_by_lambda[lambda_].append(gap(neural_len, opt_len))

    rows = []
    rows.append(("greedy distance only", float(np.mean(distance_gaps))))

    for lambda_ in LAMBDA_VALUES:
        rows.append(
            (
                f"neural greedy lambda={lambda_}",
                float(np.mean(neural_gaps_by_lambda[lambda_])),
            )
        )

    print()
    print("Tour construction results:")
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
