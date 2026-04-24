import numpy as np


def predict_all_zeros(graph) -> np.ndarray:
    """
    graph.y has shape [E]
    Returns prediction vector of the same shape, all zeros.
    """
    return np.zeros_like(graph.y, dtype=np.float32)