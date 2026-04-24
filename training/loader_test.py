from __future__ import annotations

import sys
from pathlib import Path

from torch_geometric.loader import DataLoader

# Allow running as:
# python training/loader_test.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset_generator import build_dataset
from training.torch_dataset import to_pyg


def build_pyg_dataset(num_samples: int, n: int, seed_offset: int):
    raw_dataset = build_dataset(
        num_samples=num_samples,
        n=n,
        seed_offset=seed_offset,
    )

    pyg_dataset = [to_pyg(sample) for sample in raw_dataset]
    return pyg_dataset


def main() -> None:
    n = 10

    train_dataset = build_pyg_dataset(
        num_samples=1000,
        n=n,
        seed_offset=0,
    )

    val_dataset = build_pyg_dataset(
        num_samples=200,
        n=n,
        seed_offset=10000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )

    print(f"Train graphs: {len(train_dataset)}")
    print(f"Val graphs:   {len(val_dataset)}")
    print()

    batch = next(iter(train_loader))

    print("One train batch:")
    print(f"x shape:          {batch.x.shape}")
    print(f"edge_index shape: {batch.edge_index.shape}")
    print(f"edge_attr shape:  {batch.edge_attr.shape}")
    print(f"y shape:          {batch.y.shape}")
    print(f"batch shape:      {batch.batch.shape}")
    print(f"num graphs:       {batch.num_graphs}")


if __name__ == "__main__":
    main()
