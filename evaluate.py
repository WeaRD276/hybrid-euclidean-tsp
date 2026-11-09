"""
Tour-level and edge-level evaluation for one problem size.

Example:
    python evaluate.py --n 20

Compares, on the test split:
  * classical heuristics: nearest neighbour, greedy edge, farthest insertion, (+ 2-opt)
  * hybrid greedy with non-learned heatmaps (kNN rank, SoftDist)
  * hybrid greedy with every trained run in outputs/runs/n<N>_*
For each heatmap, the fusion weight lambda and the score mode (raw / norm) are chosen
on the validation split and then applied once to the test split.

Writes results/n_<N>/{tours.md, tours.csv, edges.md, edges.csv, lambda_sweep.csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import OUTPUTS_DIR, PROJECT_ROOT
from data.graphs import tour_adjacency
from data.instances import load_split
from data.solvers import distance_matrix, tour_length
from experiments.metrics import (
    average_precision,
    best_f1_threshold,
    bootstrap_ci,
    edge_scores_and_labels,
    paired_comparison,
    precision_at_n,
    prf,
    recall_at_k,
    top_n_length_ratio,
)
from heuristics.classical import farthest_insertion, greedy_edge, nearest_neighbour, two_opt
from heuristics.scores import fused_score, knn_rank_heatmap, softdist_heatmap

LAMBDAS = {
    "raw": [0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0],
    "norm": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 128.0],
}


class Split:
    def __init__(self, n: int, name: str):
        data = load_split(n, name)
        self.points = data["points"]
        self.ref_lengths = data["lengths"]
        self.dists = [distance_matrix(p) for p in self.points]
        self.adjs = [tour_adjacency(t) for t in data["tours"]]

    def gaps(self, tours) -> np.ndarray:
        lengths = np.array([tour_length(d, t) for d, t in zip(self.dists, tours)])
        return 100.0 * (lengths / self.ref_lengths - 1.0)


def timed(fn, items):
    start = time.time()
    out = [fn(*x) for x in items]
    return out, 1000.0 * (time.time() - start) / len(items)


def hybrid_tours(split: Split, heats, lam: float, mode: str):
    return [greedy_edge(d, fused_score(d, h, lam, mode)) for d, h in zip(split.dists, heats)]


def tune_lambda(val: Split, heats, label: str, sweep_rows: list) -> tuple[str, float, float]:
    best = None
    for mode, grid in LAMBDAS.items():
        for lam in grid:
            gap = float(val.gaps(hybrid_tours(val, heats, lam, mode)).mean())
            sweep_rows.append({"heatmap": label, "mode": mode, "lambda": lam, "val_gap_pct": gap})
            if best is None or gap < best[2] - 1e-12:
                best = (mode, lam, gap)
    return best


def load_runs(n: int, pattern: str) -> list[dict]:
    runs = []
    for run_dir in sorted((OUTPUTS_DIR / "runs").glob(f"n{n}_{pattern}")):
        if not (run_dir / "heatmaps.npz").exists():
            continue
        config = json.loads((run_dir / "config.json").read_text())
        with np.load(run_dir / "heatmaps.npz") as f:
            config["val_heats"], config["test_heats"] = f["val"], f["test"]
        runs.append(config)
    return runs


def edge_metrics(val: Split, test: Split, val_heats, test_heats, probabilistic: bool) -> dict:
    scores, labels = edge_scores_and_labels(test_heats, test.adjs)
    out = {
        "AP": average_precision(labels, scores),
        "P@n": precision_at_n(test_heats, test.adjs),
        "R@2": recall_at_k(test_heats, test.adjs, 2),
        "R@5": recall_at_k(test_heats, test.adjs, 5),
        "len_ratio": top_n_length_ratio(test_heats, test.adjs, test.dists),
    }
    if probabilistic:
        val_scores, val_labels = edge_scores_and_labels(val_heats, val.adjs)
        thr = best_f1_threshold(val_labels, val_scores)
        out |= {"threshold": thr} | prf(labels, scores, thr)
    return out


def fmt_p(p: float) -> str:
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--runs", default="*", help="glob for run names after the n<N>_ prefix")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or PROJECT_ROOT / "results" / f"n_{args.n}"
    out_dir.mkdir(parents=True, exist_ok=True)

    val, test = Split(args.n, "val"), Split(args.n, "test")
    print(f"n={args.n}: {len(val.points)} val, {len(test.points)} test instances")

    # ---------------- classical baselines ----------------
    tour_rows = []

    def add_row(method, family, tours, ms, setting="", reference=None):
        gaps = test.gaps(tours)
        row = {"method": method, "family": family, "setting": setting, "gaps": gaps, "ms": ms, "reference": reference}
        tour_rows.append(row)
        return row

    items = [(d,) for d in test.dists]
    greedy_t, ms = timed(greedy_edge, items)
    greedy_row = add_row("greedy edge", "classical", greedy_t, ms)
    nn_t, ms = timed(nearest_neighbour, items)
    add_row("nearest neighbour", "classical", nn_t, ms, reference=greedy_row)
    fi_t, ms = timed(farthest_insertion, items)
    add_row("farthest insertion", "classical", fi_t, ms, reference=greedy_row)
    g2_t, ms2 = timed(two_opt, list(zip(test.dists, greedy_t)))
    greedy_2opt_row = add_row("greedy edge + 2-opt", "classical + 2-opt", g2_t, ms2, reference=greedy_row)
    fi2_t, ms2 = timed(two_opt, list(zip(test.dists, fi_t)))
    add_row("farthest insertion + 2-opt", "classical + 2-opt", fi2_t, ms2, reference=greedy_2opt_row)

    # ---------------- heatmaps ----------------
    heatmaps = [
        {"label": "kNN rank (no learning)", "group": "kNN rank", "val_heats": [knn_rank_heatmap(d) for d in val.dists],
         "test_heats": [knn_rank_heatmap(d) for d in test.dists], "probabilistic": False},
        {"label": "SoftDist (no learning)", "group": "SoftDist", "val_heats": [softdist_heatmap(d) for d in val.dists],
         "test_heats": [softdist_heatmap(d) for d in test.dists], "probabilistic": False},
    ]
    for run in load_runs(args.n, args.runs):
        heatmaps.append(run | {"label": run["name"], "group": run["label"], "probabilistic": True})

    sweep_rows, edge_rows = [], []
    distance_heats = (
        [1.0 - d / d.max() for d in val.dists],
        [1.0 - d / d.max() for d in test.dists],
    )
    edge_rows.append({"heatmap": "distance only (n shortest edges)", "group": "distance"}
                     | edge_metrics(val, test, *distance_heats, probabilistic=False))

    for hm in heatmaps:
        print(f"  tuning {hm['label']} ...", flush=True)
        mode, lam, val_gap = tune_lambda(val, hm["val_heats"], hm["label"], sweep_rows)
        tours, ms = timed(lambda d, h: greedy_edge(d, fused_score(d, h, lam, mode)),
                          list(zip(test.dists, hm["test_heats"])))
        setting = f"{mode}, λ={lam:g}"
        family = "neural hybrid" if hm["probabilistic"] else "heatmap hybrid"
        row = add_row(f"greedy + {hm['label']}", family, tours, ms, setting, reference=greedy_row)
        row["group"] = hm["group"]
        tours2, ms2 = timed(two_opt, list(zip(test.dists, tours)))
        row2 = add_row(f"greedy + {hm['label']} + 2-opt", family + " + 2-opt", tours2, ms + ms2, setting,
                       reference=greedy_2opt_row)
        row2["group"] = hm["group"]
        edge_rows.append({"heatmap": hm["label"], "group": hm["group"]}
                         | edge_metrics(val, test, hm["val_heats"], hm["test_heats"], hm["probabilistic"]))

    # ---------------- statistics and output ----------------
    for row in tour_rows:
        row["mean_gap"] = float(row["gaps"].mean())
        row["ci"] = bootstrap_ci(row["gaps"])
        if row["reference"] is not None:
            row |= paired_comparison(row["gaps"], row["reference"]["gaps"])
            row["vs"] = row["reference"]["method"]

    with open(out_dir / "tours.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["method", "family", "setting", "mean_gap", "ci_low", "ci_high", "ms", "vs", "wins", "ties", "losses", "wilcoxon_p"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in tour_rows:
            writer.writerow({k: r.get(k, "") for k in fields} | {"ci_low": r["ci"][0], "ci_high": r["ci"][1]})

    with open(out_dir / "edges.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["heatmap", "group", "AP", "P@n", "R@2", "R@5", "len_ratio", "threshold", "precision", "recall", "f1"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fields} for r in edge_rows])

    with open(out_dir / "lambda_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["heatmap", "mode", "lambda", "val_gap_pct"])
        writer.writeheader()
        writer.writerows(sweep_rows)

    write_markdown(out_dir, args.n, len(test.points), tour_rows, edge_rows)
    print(f"Saved results to {out_dir}")


def write_markdown(out_dir: Path, n: int, num_test: int, tour_rows, edge_rows) -> None:
    lines = [f"# Tour quality, n = {n} ({num_test} test instances)", ""]
    lines += [
        "Gap is relative to the LKH reference tour (optimal at this size in practice). "
        "λ and the score mode are chosen on the validation split. "
        "W/T/L and the Wilcoxon p-value compare each method per instance against the row named in “vs”.",
        "",
        "| method | setting | mean gap % [95% CI] | ms / instance | vs | W / T / L | p |",
        "| --- | --- | --- | ---: | --- | --- | ---: |",
    ]
    for r in tour_rows:
        ci = f"{r['mean_gap']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"
        cmp = (f"{r['vs']} | {r['wins']} / {r['ties']} / {r['losses']} | {fmt_p(r['wilcoxon_p'])}"
               if r["reference"] is not None else " |  | ")
        lines.append(f"| {r['method']} | {r['setting']} | {ci} | {r['ms']:.2f} | {cmp} |")

    # seed-aggregated view of the neural runs
    groups = defaultdict(list)
    for r in tour_rows:
        if r["family"].startswith("neural"):
            groups[(r["group"], r["family"])].append(r["mean_gap"])
    if groups:
        lines += ["", "## Neural runs aggregated over seeds", "",
                  "| configuration | variant | seeds | mean gap % (mean ± std over seeds) |", "| --- | --- | ---: | --- |"]
        for (group, family), gaps in sorted(groups.items()):
            variant = "+ 2-opt" if "2-opt" in family else "greedy"
            lines.append(f"| {group} | {variant} | {len(gaps)} | {np.mean(gaps):.2f} ± {np.std(gaps):.2f} |")

    lines += ["", f"# Edge prediction, n = {n}", "",
              "Computed over undirected pairs; p(u,v) and p(v,u) are averaged. "
              "AP = average precision. P@n = share of tour edges among the n top-scoring pairs. "
              "R@k = share of each node's two tour neighbours inside its top-k. "
              "Length ratio = mean length of the n top pairs ÷ mean tour-edge length (below 1 = biased to short edges). "
              "F1 uses the threshold that maximises F1 on validation (neural runs only).", "",
              "| heatmap | AP | P@n | R@2 | R@5 | length ratio | thr | P | R | F1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in edge_rows:
        tail = (f"{r['threshold']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f}"
                if "f1" in r else " |  |  | ")
        lines.append(f"| {r['heatmap']} | {r['AP']:.3f} | {r['P@n']:.3f} | {r['R@2']:.3f} | {r['R@5']:.3f} | "
                     f"{r['len_ratio']:.3f} | {tail} |")

    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
