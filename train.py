"""
Train an edge classifier and save its heatmaps for evaluation.

Example:
    python train.py --n 20 --model graphconv --graph knn --k 10 --seed 0

Writes outputs/runs/<name>/ with config.json, history.csv, model.pt and
heatmaps.npz (dense symmetric edge probabilities for the val and test splits).
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from config import OUTPUTS_DIR
from data.graphs import build_graphs, tour_adjacency
from data.instances import load_split
from data.solvers import distance_matrix
from experiments.metrics import average_precision, edge_scores_and_labels
from heuristics.scores import prob_matrix
from models.edge_gnn import EdgeGNN
from models.gat_edge_gnn import GATEdgeGNN

MODELS = {
    "graphconv": EdgeGNN,
    "gat": GATEdgeGNN,
}

RUNS_DIR = OUTPUTS_DIR / "runs"


def run_name(args) -> str:
    graph = f"knn{args.k}" if args.graph == "knn" else "complete"
    size = f"_t{args.train_size}" if args.train_size else ""
    epochs = f"_e{args.epochs}" if args.epochs != 30 else ""
    return f"n{args.n}_{args.model}_{graph}_h{args.hidden}{size}{epochs}_s{args.seed}"


def config_label(args) -> str:
    graph = f"kNN k={args.k}" if args.graph == "knn" else "complete"
    size = f", {args.train_size} train" if args.train_size else ""
    epochs = f", {args.epochs} epochs" if args.epochs != 30 else ""
    return f"{args.model}, {graph}{size}{epochs}"


@torch.no_grad()
def predict_heatmaps(model, graphs, n: int, batch_size: int = 64) -> np.ndarray:
    model.eval()
    heats = []
    for batch in DataLoader(graphs, batch_size=batch_size):
        probs = torch.sigmoid(model(batch)).numpy()
        offsets = batch.ptr[:-1].numpy()
        edge_batch = batch.batch[batch.edge_index[0]].numpy()
        ei = batch.edge_index.numpy()
        for g, off in enumerate(offsets):
            sel = edge_batch == g
            heats.append(prob_matrix(n, ei[:, sel] - off, probs[sel]))
    return np.stack(heats)


def heatmap_ap(heats: np.ndarray, split: dict) -> float:
    adjs = [tour_adjacency(t) for t in split["tours"]]
    scores, labels = edge_scores_and_labels(heats, adjs)
    return average_precision(labels, scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), default="graphconv")
    parser.add_argument("--graph", choices=["complete", "knn"], default="complete")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-size", type=int, default=None, help="use only the first N training instances")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    name = run_name(args)
    out_dir = RUNS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {s: load_split(args.n, s) for s in ("train", "val", "test")}
    if args.train_size:
        splits["train"] = {key: v[: args.train_size] for key, v in splits["train"].items()}
    graphs = {s: build_graphs(splits[s], args.graph, args.k) for s in splits}

    # Class balance depends on the graph type, so pos_weight is taken from the data.
    y = torch.cat([g.y for g in graphs["train"]])
    pos_weight = (y.numel() - y.sum()) / y.sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = MODELS[args.model](hidden_dim=args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(graphs["train"], batch_size=args.batch_size, shuffle=True)

    history = []
    best_ap = -1.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch.y)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.num_graphs

        val_ap = heatmap_ap(predict_heatmaps(model, graphs["val"], args.n), splits["val"])
        history.append({"epoch": epoch, "train_loss": total / len(graphs["train"]), "val_ap": val_ap})
        print(f"[{name}] epoch {epoch:02d} loss={history[-1]['train_loss']:.4f} val_AP={val_ap:.4f}", flush=True)

        if val_ap > best_ap:
            best_ap = val_ap
            torch.save(model.state_dict(), out_dir / "model.pt")

    train_seconds = time.time() - start
    model.load_state_dict(torch.load(out_dir / "model.pt"))

    infer_start = time.time()
    test_heats = predict_heatmaps(model, graphs["test"], args.n)
    infer_ms = 1000 * (time.time() - infer_start) / len(graphs["test"])

    np.savez_compressed(
        out_dir / "heatmaps.npz",
        val=predict_heatmaps(model, graphs["val"], args.n).astype(np.float32),
        test=test_heats.astype(np.float32),
    )

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    config = vars(args) | {
        "name": name,
        "label": config_label(args),
        "best_val_ap": best_ap,
        "best_epoch": max(history, key=lambda h: h["val_ap"])["epoch"],
        "train_instances": len(graphs["train"]),
        "train_seconds": round(train_seconds, 1),
        "inference_ms_per_instance": round(infer_ms, 3),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[{name}] done: best val AP {best_ap:.4f}, {train_seconds:.0f}s")


if __name__ == "__main__":
    main()
