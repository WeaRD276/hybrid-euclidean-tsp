from pathlib import Path
import json

from data.generator import gen_points
from data.graph_builder import TSPGraphData
from data.serialization import save_sample
from config import OUTPUTS_DIR, SAVED_DATA_DIR


def build_dataset(num_samples: int, n: int, seed_offset: int = 0):
    res = []
    for i in range(num_samples):
        points = gen_points(seed_offset + i, n)
        res.append(TSPGraphData.from_points(points))
    return res


def save_metadata(
    n: int,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    train_seed_offset: int,
    val_seed_offset: int,
    test_seed_offset: int,
) -> None:
    base_dir = SAVED_DATA_DIR / f"n_{n}"
    base_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "n": n,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "train_seed_offset": train_seed_offset,
        "val_seed_offset": val_seed_offset,
        "test_seed_offset": test_seed_offset,
        "node_features": ["x", "y"],
        "edge_features": ["dx", "dy", "dist"],
        "label": "edge_in_optimal_tour",
    }

    metadata_path = base_dir / "meta.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def generate_split(
    out_dir: str,
    num_samples: int,
    n: int,
    seed_offset: int = 0,
) -> None:
    out_path = SAVED_DATA_DIR / f"n_{n}" / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        points = gen_points(seed_offset + i, n)
        sample = TSPGraphData.from_points(points)
        file_path = out_path / f"sample_{i:06d}.pkl"

        save_sample(sample, file_path)


def generate_all_splits(n) -> None:
    train_samples = 500
    val_samples = 100
    test_samples = 100

    train_seed_offset = 0
    val_seed_offset = 10000
    test_seed_offset = 20000

    generate_split("train", num_samples=train_samples, n=n, seed_offset=train_seed_offset)
    generate_split("val", num_samples=val_samples, n=n, seed_offset=val_seed_offset)
    generate_split("test", num_samples=test_samples, n=n, seed_offset=test_seed_offset)

    save_metadata(
        n=n,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        train_seed_offset=train_seed_offset,
        val_seed_offset=val_seed_offset,
        test_seed_offset=test_seed_offset,
    )


def main():
    n = 20
    generate_all_splits(n)


if __name__ == "__main__":
    main()