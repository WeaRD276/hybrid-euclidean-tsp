from pathlib import Path

import torch_geometric

from data.generator import gen_points
from data.graph_builder import TSPGraphData
from data.serialization import save_sample

def build_dataset(num_samples: int, n: int, seed_offset: int = 0):
    res = []
    for i in range(num_samples):
        points = gen_points(seed_offset + i, n)
        res.append(TSPGraphData.from_points(points))
    return res

def generate_split(
    out_dir: str,
    num_samples: int,
    n: int,
    seed_offset: int = 0,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # dataset = build_dataset(num_samples=num_samples, n=n, seed_offset=seed_offset)
    # for i in range(num_samples):
    #     sample = dataset[i]
    #
    #     file_path = out_path / f"sample_{i:06d}.pkl"
    #     save_sample(sample, file_path)

    for i in range(num_samples):
        points = gen_points(seed_offset + i, n)
        sample = TSPGraphData.from_points(points)
        file_path = out_path / f"sample_{i:06d}.pkl"

        save_sample(sample, file_path)


def generate_all_splits() -> None:
    generate_split("outputs/datasets/train", num_samples=1000, n=10, seed_offset=0)
    generate_split("outputs/datasets/val", num_samples=200, n=10, seed_offset=10000)
    generate_split("outputs/datasets/test", num_samples=200, n=10, seed_offset=20000)


def main():
    generate_all_splits()

    # train_data = build_dataset(num_samples=1000, n=10)
    # val_data = build_dataset(num_samples=200, n=10, seed_offset=10000)
    # sample = train_data[0]
    # print(sample.x.shape)
    # print(sample.edge_index.shape)
    # print(sample.edge_attr.shape)
    # print(sample.y.shape)
    # print(sample.y.mean())


if __name__ == "__main__":
    main()
