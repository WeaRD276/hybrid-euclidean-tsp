from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from analysis.distance_analysis import run_distance_analysis
from analysis.threshold_analysis_gat import save_threshold_results
from analysis.visualization import save_graph_visualizations
from baselines.shortest_edges import predict_shortest_edges_unconstrained
from data.serialization import load_split_samples
from models.edge_gnn import make_loss_fn
from models.gat_edge_gnn import GATEdgeGNN
from training.torch_dataset import TSPPyGDataset, to_pyg
from training.train_gat_edge_classifier import (
    N,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    THRESHOLDS,
    evaluate_model,
)
from config import OUTPUTS_DIR, SAVED_DATA_DIR


THRESHOLD_OUT = OUTPUTS_DIR / "tables" / f"thresholds_gat_n{N}.csv"
DISTANCE_OUT = OUTPUTS_DIR / "reports" / f"distance_analysis_gat_n{N}.md"
PLOTS_OUT = OUTPUTS_DIR / "plots" / f"graphs_gat_n{N}"

BEST_THRESHOLD = 0.55
NUM_VISUALIZATIONS = 3

TEST_DIR = SAVED_DATA_DIR / f"n_{N}" / "test"


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

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Saved test split not found: {TEST_DIR}\n"
            "Generate dataset first:\n"
            "python data/dataset_generator.py"
        )

    print("Loading test dataset from files...")
    test_raw = load_split_samples(TEST_DIR)
    test_dataset = TSPPyGDataset(TEST_DIR)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("Loading model...")
    model = GATEdgeGNN(hidden_dim=64, heads=4, dropout=0.0).to(device)
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
    for graph_id in range(min(NUM_VISUALIZATIONS, len(test_raw))):
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
