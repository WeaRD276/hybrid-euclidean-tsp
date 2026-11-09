"""
Instance generation: uniform random points in the unit square plus a reference tour.

Each split is stored as one compressed .npz file:
    points  [num_instances, n, 2]
    tours   [num_instances, n]     node order of the reference tour
    lengths [num_instances]        reference tour length

Usage:
    python -m data.instances --n 20 --train 10000 --val 1000 --test 1000
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from config import SAVED_DATA_DIR
from data.solvers import SOLVERS, distance_matrix, tour_length

SPLIT_SEEDS = {"train": 0, "val": 1, "test": 2}


def split_path(n: int, split: str) -> Path:
    return SAVED_DATA_DIR / f"n_{n}" / f"{split}.npz"


def random_points(num_instances: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng([n, seed])
    return rng.random((num_instances, n, 2))


def _solve_one(args):
    points, solver = args
    tour = SOLVERS[solver](points)
    return tour, tour_length(distance_matrix(points), tour)


def generate_split(n: int, split: str, num_instances: int, solver: str, workers: int) -> Path:
    points = random_points(num_instances, n, SPLIT_SEEDS[split])

    start = time.time()
    with Pool(workers) as pool:
        results = pool.map(_solve_one, [(p, solver) for p in points], chunksize=64)
    elapsed = time.time() - start

    tours = np.stack([r[0] for r in results])
    lengths = np.array([r[1] for r in results])

    path = split_path(n, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=points, tours=tours, lengths=lengths)

    meta = {
        "n": n,
        "split": split,
        "num_instances": num_instances,
        "seed": SPLIT_SEEDS[split],
        "solver": solver,
        "solve_seconds": round(elapsed, 2),
        "mean_length": float(lengths.mean()),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"n={n} {split}: {num_instances} instances, solver={solver}, {elapsed:.1f}s")
    return path


def load_split(n: int, split: str) -> dict[str, np.ndarray]:
    path = split_path(n, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it with:\n"
            f"python -m data.instances --n {n}"
        )
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", required=True)
    parser.add_argument("--train", type=int, default=10000)
    parser.add_argument("--val", type=int, default=1000)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--solver", choices=sorted(SOLVERS), default="lkh")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    for n in args.n:
        for split in ("train", "val", "test"):
            generate_split(n, split, getattr(args, split), args.solver, args.workers)


if __name__ == "__main__":
    main()
