from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.all_zeros import predict_all_zeros
from baselines.shortest_edges import predict_shortest_edges_unconstrained
from data.serialization import load_split_samples
from experiments.evaluate import compute_confusion, compute_metrics
from models.edge_gnn import EdgeGNN, make_loss_fn
from training.torch_dataset import TSPPyGDataset
from config import *


N = 10

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
THRESHOLDS = [i / 100 for i in range(5, 96, 5)]

CHECKPOINT_PATH = OUTPUTS_DIR / "checkpoints" / "edge_gnn_n10_best.pt"
RESULTS_PATH = OUTPUTS_DIR / "reports" / "results_n10.md"

TRAIN_DIR = SAVED_DATA_DIR / f"n_{N}" / "train"
VAL_DIR = SAVED_DATA_DIR / f"n_{N}" / "val"
TEST_DIR = SAVED_DATA_DIR / f"n_{N}" / "test"


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0.0
    total_edges = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, batch.y)

        loss.backward()
        optimizer.step()

        num_edges = batch.y.numel()
        total_loss += loss.item() * num_edges
        total_edges += num_edges

    return total_loss / total_edges


@torch.no_grad()
def evaluate_model(model, loader, loss_fn, device, threshold: float = 0.5):
    model.eval()

    total_loss = 0.0
    total_edges = 0
    total_conf = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for batch in loader:
        batch = batch.to(device)

        logits = model(batch)
        loss = loss_fn(logits, batch.y)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long().cpu().numpy()
        targets = batch.y.long().cpu().numpy()

        conf = compute_confusion(targets, preds)
        for key in total_conf:
            total_conf[key] += conf[key]

        num_edges = batch.y.numel()
        total_loss += loss.item() * num_edges
        total_edges += num_edges

    return total_loss / total_edges, compute_metrics(total_conf), total_conf


@torch.no_grad()
def find_best_threshold(model, loader, loss_fn, device):
    best_threshold = None
    best_loss = None
    best_metrics = None
    best_conf = None
    best_f1 = -1.0

    for threshold in THRESHOLDS:
        loss, metrics, conf = evaluate_model(
            model=model,
            loader=loader,
            loss_fn=loss_fn,
            device=device,
            threshold=threshold,
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_loss = loss
            best_metrics = metrics
            best_conf = conf

    return best_threshold, best_loss, best_metrics, best_conf


def evaluate_baseline_on_raw_dataset(raw_dataset, predictor):
    total_conf = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for graph in raw_dataset:
        y_true = graph.y.astype(np.int32)
        y_pred = predictor(graph).astype(np.int32)

        assert y_true.shape == y_pred.shape

        conf = compute_confusion(y_true, y_pred)
        for key in total_conf:
            total_conf[key] += conf[key]

    return compute_metrics(total_conf)


def print_final_table(rows):
    print()
    print("Final test results:")
    print()
    print("| method           | precision | recall | F1 |")
    print("| ---------------- | --------: | -----: | --: |")

    for method, metrics in rows:
        print(
            f"| {method:<16} | "
            f"{metrics['precision']:9.6f} | "
            f"{metrics['recall']:6.6f} | "
            f"{metrics['f1']:.6f} |"
        )


def save_results_markdown(
    path,
    best_epoch,
    best_val_f1,
    best_threshold,
    val_metrics_at_threshold,
    test_loss,
    rows,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Results for n = {N}")
    lines.append("")
    lines.append(f"- Best epoch: {best_epoch}")
    lines.append(f"- Best validation F1 during training at threshold 0.5: {best_val_f1:.6f}")
    lines.append(f"- Selected validation threshold: {best_threshold:.2f}")
    lines.append(f"- Validation F1 at selected threshold: {val_metrics_at_threshold['f1']:.6f}")
    lines.append(f"- EdgeGNN test loss: {test_loss:.6f}")
    lines.append("")
    lines.append("| method | precision | recall | F1 |")
    lines.append("| --- | ---: | ---: | ---: |")

    for method, metrics in rows:
        lines.append(
            f"| {method} | "
            f"{metrics['precision']:.6f} | "
            f"{metrics['recall']:.6f} | "
            f"{metrics['f1']:.6f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    torch.manual_seed(47)
    np.random.seed(47)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    if not TRAIN_DIR.exists() or not VAL_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            "Saved dataset splits were not found.\n"
            f"Expected:\n{TRAIN_DIR}\n{VAL_DIR}\n{TEST_DIR}\n"
            "Generate dataset first:\n"
            "python data/dataset_generator.py"
        )

    print("Loading datasets from files...")
    train_dataset = TSPPyGDataset(TRAIN_DIR)
    val_dataset = TSPPyGDataset(VAL_DIR)
    test_dataset = TSPPyGDataset(TEST_DIR)

    # raw datasets only for baseline evaluation
    val_raw = load_split_samples(VAL_DIR)
    test_raw = load_split_samples(TEST_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train graphs: {len(train_dataset)}")
    print(f"Val graphs:   {len(val_dataset)}")
    print(f"Test graphs:  {len(test_dataset)}")
    print()

    model = EdgeGNN(hidden_dim=64).to(device)
    loss_fn = make_loss_fn(num_nodes=N, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    checkpoint_path = Path(CHECKPOINT_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("Training EdgeGNN...")
    best_val_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val_loss, val_metrics, _ = evaluate_model(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_precision={val_metrics['precision']:.6f} | "
            f"val_recall={val_metrics['recall']:.6f} | "
            f"val_f1={val_metrics['f1']:.6f}"
        )

    print()
    print(f"Best neural val F1: {best_val_f1:.6f} at epoch {best_epoch}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print()

    best_model = EdgeGNN(hidden_dim=64).to(device)
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    best_threshold, val_loss_thr, val_metrics_thr, _ = find_best_threshold(
        model=best_model,
        loader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print(
        f"Best validation threshold: {best_threshold:.2f} | "
        f"val_precision={val_metrics_thr['precision']:.6f} | "
        f"val_recall={val_metrics_thr['recall']:.6f} | "
        f"val_f1={val_metrics_thr['f1']:.6f}"
    )
    print()

    test_loss, edge_gnn_metrics, _ = evaluate_model(
        model=best_model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        threshold=best_threshold,
    )

    print(f"Best EdgeGNN test loss: {test_loss:.6f}")

    all_zeros_metrics = evaluate_baseline_on_raw_dataset(
        raw_dataset=test_raw,
        predictor=predict_all_zeros,
    )

    shortest_edges_metrics = evaluate_baseline_on_raw_dataset(
        raw_dataset=test_raw,
        predictor=predict_shortest_edges_unconstrained,
    )

    final_rows = [
        ("all zeros", all_zeros_metrics),
        ("n shortest edges", shortest_edges_metrics),
        ("EdgeGNN", edge_gnn_metrics),
    ]

    print_final_table(final_rows)

    save_results_markdown(
        path=RESULTS_PATH,
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        best_threshold=best_threshold,
        val_metrics_at_threshold=val_metrics_thr,
        test_loss=test_loss,
        rows=final_rows,
    )

    print()
    print(f"Saved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
