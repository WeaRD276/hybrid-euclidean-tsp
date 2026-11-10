"""
Hybridization mechanisms: the same heatmaps used in different ways (WP5).

Example:
    python -m experiments.mechanisms --n 50 --runs "graphconv_knn10_h64_s*"

For every heatmap (distance kNN, kNN rank, SoftDist, each neural run) and on the test split:
  fused greedy        greedy on d - λ·p, λ and mode from results/n_<N>/lambda_sweep.csv
  pruned greedy (k)   fused greedy where only each node's top-k heatmap edges may be taken
                      first; every other edge comes after them, ordered by distance
  2-opt cand (k)      neighbour-list 2-opt from the plain greedy tour, using only moves that
                      add one of each node's top-k heatmap edges
  fused + full 2-opt  ordinary 2-opt from the fused greedy tour
  fused + 2-opt cand  the same neighbour-list 2-opt starting from the fused greedy tour

Writes results/n_<N>/mechanisms.md and mechanisms.csv.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT
from evaluate import Split, load_runs
from experiments.metrics import bootstrap_ci, paired_comparison
from heuristics.classical import candidate_mask, greedy_edge, two_opt
from heuristics.scores import fused_score, knn_rank_heatmap, softdist_heatmap

PRUNE_K = [2, 3, 5]
TWO_OPT_K = [2, 3, 5, 8]
MECHANISM_ORDER = ["fused greedy", "pruned greedy", "fused + full 2-opt", "greedy + 2-opt cand", "fused + 2-opt cand"]
NOT_CANDIDATE_PENALTY = 1e6


def tuned_setting(sweep_path: Path, label: str) -> tuple[str, float]:
    rows = [r for r in csv.DictReader(open(sweep_path)) if r["heatmap"] == label]
    if not rows:
        return "raw", 1000.0
    best = min(rows, key=lambda r: float(r["val_gap_pct"]))
    return best["mode"], float(best["lambda"])


def pruned_greedy(dist, heat, k, lam, mode):
    score = fused_score(dist, heat, lam, mode)
    score = score + NOT_CANDIDATE_PENALTY * ~candidate_mask(heat, k)
    return greedy_edge(dist, score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--runs", default="*", help="glob for run names after the n<N>_ prefix")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "results" / f"n_{args.n}"
    sweep_path = out_dir / "lambda_sweep.csv"
    test = Split(args.n, "test")
    dists = test.dists
    greedy_tours = [greedy_edge(d) for d in dists]

    heatmaps = [
        ("distance (k nearest)", "distance", [1.0 - d / d.max() for d in dists], None),
        ("kNN rank (no learning)", "kNN rank", [knn_rank_heatmap(d) for d in dists], "kNN rank (no learning)"),
        ("SoftDist (no learning)", "SoftDist", [softdist_heatmap(d) for d in dists], "SoftDist (no learning)"),
    ]
    for run in load_runs(args.n, args.runs):
        heatmaps.append((run["name"], run["label"], run["test_heats"], run["name"]))

    rows = []
    reference_gaps = {"greedy edge": test.gaps(greedy_tours),
                      "greedy edge + full 2-opt": test.gaps([two_opt(d, t) for d, t in zip(dists, greedy_tours)])}

    for label, group, heats, sweep_label in heatmaps:
        print(f"  {label} ...", flush=True)
        mode, lam = tuned_setting(sweep_path, sweep_label) if sweep_label else ("raw", 0.0)

        def add(mechanism, k, tours):
            rows.append({"heatmap": label, "group": group, "mechanism": mechanism, "k": k,
                         "setting": f"{mode}, λ={lam:g}", "gaps": test.gaps(tours)})

        fused = [greedy_edge(d, fused_score(d, h, lam, mode)) for d, h in zip(dists, heats)]
        if sweep_label:
            add("fused greedy", "", fused)
            add("fused + full 2-opt", "", [two_opt(d, t) for d, t in zip(dists, fused)])
            for k in PRUNE_K:
                add("pruned greedy", k, [pruned_greedy(d, h, k, lam, mode) for d, h in zip(dists, heats)])
        for k in TWO_OPT_K:
            cands = [candidate_mask(h, k) for h in heats]
            add("greedy + 2-opt cand", k,
                [two_opt(d, t, candidates=c) for d, t, c in zip(dists, greedy_tours, cands)])
            if sweep_label:
                add("fused + 2-opt cand", k,
                    [two_opt(d, t, candidates=c) for d, t, c in zip(dists, fused, cands)])

    # Every mechanism's neural rows are compared per instance with the SoftDist row
    # of the same mechanism and k, the strongest non-learned heatmap.
    softdist = {(r["mechanism"], r["k"]): r["gaps"] for r in rows if r["group"] == "SoftDist"}
    for r in rows:
        r["mean_gap"] = float(r["gaps"].mean())
        r["ci"] = bootstrap_ci(r["gaps"])
        ref = softdist.get((r["mechanism"], r["k"]))
        if ref is not None and r["group"] not in ("SoftDist", "distance", "kNN rank"):
            r |= paired_comparison(r["gaps"], ref)

    with open(out_dir / "mechanisms.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["heatmap", "group", "mechanism", "k", "setting", "mean_gap", "ci_low", "ci_high",
                  "wins", "ties", "losses", "wilcoxon_p"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields} | {"ci_low": r["ci"][0], "ci_high": r["ci"][1]})

    write_markdown(out_dir, args.n, len(dists), rows, reference_gaps)
    print(f"Saved {out_dir / 'mechanisms.md'}")


def write_markdown(out_dir: Path, n: int, num_test: int, rows, reference_gaps) -> None:
    mechanisms = sorted(dict.fromkeys((r["mechanism"], r["k"]) for r in rows),
                        key=lambda m: (MECHANISM_ORDER.index(m[0]), str(m[1])))
    groups = list(dict.fromkeys(r["group"] for r in rows))

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["group"], r["mechanism"], r["k"])].append(r)

    lines = [f"# Hybridization mechanisms, n = {n} ({num_test} test instances)", "",
             "Mean test gap % over LKH; neural columns are the mean ± std over seeds. "
             "k = candidate edges per node taken from the heatmap. "
             "“vs SoftDist” sums per-instance wins / ties / losses of the neural runs against SoftDist "
             "with the same mechanism and k, and gives the largest Wilcoxon p over seeds.", "",
             "Reference: " + ", ".join(f"{name} {g.mean():.2f}%" for name, g in reference_gaps.items()), ""]

    header = "| mechanism | k | " + " | ".join(groups) + " | vs SoftDist (W / T / L, max p) |"
    lines += [header, "|" + " --- |" * (len(groups) + 3)]
    for mech, k in mechanisms:
        cells = []
        cmp = ""
        for g in groups:
            cell = by_cell.get((g, mech, k), [])
            if not cell:
                cells.append("")
                continue
            gaps = [r["mean_gap"] for r in cell]
            cells.append(f"{np.mean(gaps):.2f}" + (f" ± {np.std(gaps):.2f}" if len(cell) > 1 else ""))
            if "wins" in cell[0]:
                w = sum(r["wins"] for r in cell)
                t = sum(r["ties"] for r in cell)
                l = sum(r["losses"] for r in cell)
                p = max(r["wilcoxon_p"] for r in cell)
                cmp = f"{w} / {t} / {l}, p ≤ {p:.2g}" if p >= 1e-4 else f"{w} / {t} / {l}, p < 1e-4"
        lines.append(f"| {mech} | {k} | " + " | ".join(cells) + f" | {cmp} |")

    (out_dir / "mechanisms.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
