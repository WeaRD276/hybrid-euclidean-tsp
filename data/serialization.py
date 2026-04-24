import pickle
from pathlib import Path

def save_sample(sample, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(sample, f)

def load_sample(path):
    with open(path, "rb") as f:
        return pickle.load(f)