from pathlib import Path
import pickle


def save_sample(sample, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(sample, f)


def load_sample(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_split_samples(split_dir: str | Path):
    split_dir = Path(split_dir)
    files = sorted(split_dir.glob("*.pkl"))
    return [load_sample(path) for path in files]
