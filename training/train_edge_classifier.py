from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Allow running as:
# python training/train_edge_classifier.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.all_zeros import predict_all_zeros
from baselines.shortest_edges import predict_shortest_edges_unconstrained
from data.dataset_generator import build_dataset
from experiments.evaluate import compute_confusion, compute_metrics
from models.edge_gnn import EdgeGNN, make_loss_fn
from training.torch_dataset import to_pyg


N = 10
TRAIN_GRAPHS = 1000
VAL_GRAPHS = 200
TRAIN_SEED_OFFSET = 0
VAL_SEED_OFFSET = 10000

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
THRESHOLD = 0.5


def build_pyg_dataset(num_samples: int, n: int, seed_offset: int):
    raw_dataset = build_dataset(
        num_samples=num_samples,
        n=n,
        seed_offset=seed_offset,
    )
    pyg_dataset = [to_pyg(sample) for sample in raw_dataset]
    return raw_dataset, pyg_dataset


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
def evaluate_model(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_edges = 0

    total_conf = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for batch in loader:
        batch = batch.to(device)

        logits = model(batch)
        loss = loss_fn(logits, batch.y)

        probs = torch.sigmoid(logits)
        preds = (probs >= THRESHOLD).long().cpu().numpy()
        targets = batch.y.long().cpu().numpy()

        conf = compute_confusion(targets, preds)
        for key in total_conf:
            total_conf[key] += conf[key]

        num_edges = batch.y.numel()
        total_loss += loss.item() * num_edges
        total_edges += num_edges

    metrics = compute_metrics(total_conf)
    avg_loss = total_loss / total_edges

    return avg_loss, metrics, total_conf


def evaluate_baseline_on_raw_dataset(raw_dataset, predictor, name: str):
    total_conf = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for graph in raw_dataset:
        y_true = graph.y.astype(np.int32)
        y_pred = predictor(graph).astype(np.int32)

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

    return metrics


def main():
    torch.manual_seed(47)
    np.random.seed(47)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    print("Building datasets...")
    train_raw, train_dataset = build_pyg_dataset(
        num_samples=TRAIN_GRAPHS,
        n=N,
        seed_offset=TRAIN_SEED_OFFSET,
    )
    val_raw, val_dataset = build_pyg_dataset(
        num_samples=VAL_GRAPHS,
        n=N,
        seed_offset=VAL_SEED_OFFSET,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"Train graphs: {len(train_dataset)}")
    print(f"Val graphs:   {len(val_dataset)}")
    print()

    model = EdgeGNN(hidden_dim=64).to(device)
    loss_fn = make_loss_fn(num_nodes=N, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
    print()

    print("Baselines on the same validation dataset:")
    evaluate_baseline_on_raw_dataset(
        raw_dataset=val_raw,
        predictor=predict_all_zeros,
        name="All zeros",
    )
    evaluate_baseline_on_raw_dataset(
        raw_dataset=val_raw,
        predictor=predict_shortest_edges_unconstrained,
        name="n shortest undirected edges",
    )


if __name__ == "__main__":
    main()
