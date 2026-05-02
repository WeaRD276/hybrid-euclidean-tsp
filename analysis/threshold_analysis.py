from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from data.serialization import load_split_samples
from models.edge_gnn import EdgeGNN, make_loss_fn
from training.torch_dataset import TSPPyGDataset
from training.train_edge_classifier import (
    N,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    THRESHOLDS,
    evaluate_model,
)
from config import OUTPUTS_DIR, SAVED_DATA_DIR


OUT_PATH = OUTPUTS_DIR / "tables" / f"thresholds_n{N}.csv"
TEST_DIR = SAVED_DATA_DIR / f"n_{N}" / "test"


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
    test_dataset = TSPPyGDataset(TEST_DIR)

    # Optional sanity check that files are readable as raw samples too.
    _ = load_split_samples(TEST_DIR)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = EdgeGNN(hidden_dim=64).to(device)
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
