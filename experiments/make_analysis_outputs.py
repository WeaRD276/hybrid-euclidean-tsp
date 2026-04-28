from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.distance_analysis import run_distance_analysis
from analysis.threshold_analysis import save_threshold_results
from analysis.visualization import save_graph_visualizations
from baselines.shortest_edges import predict_shortest_edges_unconstrained
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
from training.torch_dataset import to_pyg


THRESHOLD_OUT = ROOT / "outputs" / "tables" / "thresholds_n10.csv"
DISTANCE_OUT = ROOT / "outputs" / "reports" / "distance_analysis_n10.md"
PLOTS_OUT = ROOT / "outputs" / "plots" / "graphs"

BEST_THRESHOLD = 0.60
NUM_VISUALIZATIONS = 3


@torch.no_grad()
def predict_model_mask(model, graph, device, threshold: float) -> np.ndarray:
    model.eval()

    data = to_pyg(graph).to(device)
    logits = model(data)
    probs = torch.sigmoid(logits)

    return (probs >= threshold).cpu().numpy().astype(np.float32)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run training first:\n"
            "python training/train_edge_classifier.py"
        )

    print("Building test dataset...")
    test_raw, test_dataset = build_pyg_dataset(
        num_samples=TEST_GRAPHS,
        n=N,
        seed_offset=TEST_SEED_OFFSET,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("Loading model...")
    model = EdgeGNN(hidden_dim=64).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    loss_fn = make_loss_fn(num_nodes=N, device=device)

    print("Saving threshold table...")
    threshold_rows = []
    for threshold in THRESHOLDS:
        _, metrics, _ = evaluate_model(
            model=model,
            loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            threshold=threshold,
        )

        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    save_threshold_results(
        path=THRESHOLD_OUT,
        rows=threshold_rows,
    )

    print("Computing predictions for distance analysis and visualizations...")
    baseline_predictions = []
    model_predictions = []

    for graph in test_raw:
        baseline_predictions.append(
            predict_shortest_edges_unconstrained(graph)
        )
        model_predictions.append(
            predict_model_mask(
                model=model,
                graph=graph,
                device=device,
                threshold=BEST_THRESHOLD,
            )
        )

    print("Saving distance analysis...")
    run_distance_analysis(
        graphs=test_raw,
        baseline_predictions=baseline_predictions,
        model_predictions=model_predictions,
        out_path=DISTANCE_OUT,
    )

    print("Saving visualizations...")
    for graph_id in range(NUM_VISUALIZATIONS):
        save_graph_visualizations(
            graph=test_raw[graph_id],
            pred_mask=model_predictions[graph_id],
            out_dir=PLOTS_OUT,
            graph_id=graph_id,
        )

    print("Done.")
    print(f"Thresholds: {THRESHOLD_OUT}")
    print(f"Distance analysis: {DISTANCE_OUT}")
    print(f"Plots: {PLOTS_OUT}")


if __name__ == "__main__":
    main()
