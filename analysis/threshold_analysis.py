from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.edge_gnn import EdgeGNN, make_loss_fn
from training.train_edge_classifier import (
    N,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    TEST_GRAPHS,
    TEST_SEED_OFFSET,
    THRESHOLDS,
    build_pyg_dataset,
    evaluate_model,
)


OUT_PATH = "experiments/thresholds_n10.csv"


def save_threshold_results(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "precision", "recall", "f1"],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "threshold": f"{row['threshold']:.2f}",
                    "precision": f"{row['precision']:.6f}",
                    "recall": f"{row['recall']:.6f}",
                    "f1": f"{row['f1']:.6f}",
                }
            )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_dataset = build_pyg_dataset(
        num_samples=TEST_GRAPHS,
        n=N,
        seed_offset=TEST_SEED_OFFSET,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = EdgeGNN(hidden_dim=64).to(device)

    checkpoint_path = Path(CHECKPOINT_PATH)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run training first:\n"
            "python training/train_edge_classifier.py"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    loss_fn = make_loss_fn(num_nodes=N, device=device)

    rows = []

    for threshold in THRESHOLDS:
        _, metrics, _ = evaluate_model(
            model=model,
            loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            threshold=threshold,
        )

        rows.append(
            {
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    save_threshold_results(
        path=OUT_PATH,
        rows=rows,
    )

    print(f"Saved threshold analysis to: {OUT_PATH}")


if __name__ == "__main__":
    main()
