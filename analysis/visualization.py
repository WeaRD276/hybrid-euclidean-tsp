from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def directed_to_undirected_edges(
    edge_index: np.ndarray,
    mask: np.ndarray,
) -> set[tuple[int, int]]:
    edges = set()

    for k in range(edge_index.shape[1]):
        if not mask[k]:
            continue

        u = int(edge_index[0, k])
        v = int(edge_index[1, k])

        if u == v:
            continue

        a, b = min(u, v), max(u, v)
        edges.add((a, b))

    return edges


def _draw_points(ax, points: np.ndarray) -> None:
    ax.scatter(points[:, 0], points[:, 1], s=60, zorder=4)

    for i, (x, y) in enumerate(points):
        ax.text(x + 0.01, y + 0.01, str(i), fontsize=9)


def _draw_edges(
    ax,
    points: np.ndarray,
    edges: set[tuple[int, int]],
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
    label: str | None = None,
) -> None:
    first = True

    for u, v in edges:
        x1, y1 = points[u]
        x2, y2 = points[v]

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
            label=label if first else None,
        )

        first = False


def _format_axes(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)


def plot_graph_edges(
    graph,
    edge_mask: np.ndarray,
    path: str | Path,
    title: str,
    color: str,
    linewidth: float = 2.0,
) -> None:
    points = np.asarray(graph.x)
    edges = directed_to_undirected_edges(graph.edge_index, edge_mask.astype(bool))

    fig, ax = plt.subplots(figsize=(6, 6))

    _draw_points(ax, points)
    _draw_edges(
        ax=ax,
        points=points,
        edges=edges,
        color=color,
        linewidth=linewidth,
        alpha=0.85,
        zorder=2,
    )
    _format_axes(ax, title)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_prediction_overlay(
    graph,
    pred_mask: np.ndarray,
    path: str | Path,
    title: str = "GT vs prediction",
) -> None:
    points = np.asarray(graph.x)

    gt_edges = directed_to_undirected_edges(
        graph.edge_index,
        graph.y.astype(bool),
    )
    pred_edges = directed_to_undirected_edges(
        graph.edge_index,
        pred_mask.astype(bool),
    )

    only_gt = gt_edges - pred_edges
    only_pred = pred_edges - gt_edges
    both = gt_edges & pred_edges

    fig, ax = plt.subplots(figsize=(6, 6))

    _draw_points(ax, points)

    _draw_edges(
        ax=ax,
        points=points,
        edges=only_gt,
        color="green",
        linewidth=2.0,
        alpha=0.75,
        zorder=2,
        label="GT only",
    )

    _draw_edges(
        ax=ax,
        points=points,
        edges=only_pred,
        color="red",
        linewidth=2.0,
        alpha=0.75,
        zorder=2,
        label="Prediction only",
    )

    _draw_edges(
        ax=ax,
        points=points,
        edges=both,
        color="black",
        linewidth=4.0,
        alpha=0.90,
        zorder=3,
        label="Intersection",
    )

    _format_axes(ax, title)
    ax.legend(loc="upper right")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_graph_visualizations(
    graph,
    pred_mask: np.ndarray,
    out_dir: str | Path,
    graph_id: int,
) -> None:
    out_dir = Path(out_dir)
    prefix = out_dir / f"graph_{graph_id:03d}"

    plot_graph_edges(
        graph=graph,
        edge_mask=graph.y,
        path=prefix.with_name(prefix.name + "_gt.png"),
        title=f"Graph {graph_id:03d}: optimal edges",
        color="green",
    )

    plot_graph_edges(
        graph=graph,
        edge_mask=pred_mask,
        path=prefix.with_name(prefix.name + "_pred.png"),
        title=f"Graph {graph_id:03d}: predicted edges",
        color="red",
    )

    plot_prediction_overlay(
        graph=graph,
        pred_mask=pred_mask,
        path=prefix.with_name(prefix.name + "_overlay.png"),
        title=f"Graph {graph_id:03d}: GT vs prediction",
    )
