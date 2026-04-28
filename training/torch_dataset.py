from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
# from torch_geometric.data import Dataset
from torch_geometric.data import Data

from data.serialization import load_sample
from data.graph_builder import TSPGraphData
from data.dataset_generator import build_dataset


def to_pyg(sample: TSPGraphData) -> Data:
    data = Data(
        x=torch.tensor(sample.x, dtype=torch.float32),
        edge_index=torch.tensor(sample.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(sample.edge_attr, dtype=torch.float32),
        y=torch.tensor(sample.y, dtype=torch.float32),
    )
    data.num_nodes = sample.x.shape[0]
    return data


class TSPPyGDataset(Dataset):
    def __init__(self, root_dir: str):
        self.files = sorted(Path(root_dir).glob("*.pkl"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        sample = load_sample(self.files[idx])
        return to_pyg(sample)

def build_pyg_dataset(num_samples: int, n: int, seed_offset: int):
    raw_dataset = build_dataset(
        num_samples=num_samples,
        n=n,
        seed_offset=seed_offset,
    )
    pyg_dataset = [to_pyg(sample) for sample in raw_dataset]
    return raw_dataset, pyg_dataset
