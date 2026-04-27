import numpy as np


def predict_shortest_edges_unconstrained(graph) -> np.ndarray:
    """
    Pick n shortest undirected edges, then mark both directions positive.
    Returns directed prediction vector of shape [E].
    """
    n = graph.x.shape[0]
    e = graph.y.shape[0]

    # edge_attr[k] = [dx, dy, dist]
    dists = graph.edge_attr[:, 2]
    edge_index = graph.edge_index

    undirected = []
    for k in range(e):
        u = int(edge_index[0, k])
        v = int(edge_index[1, k])
        if u < v:
            undirected.append((dists[k], u, v))

    undirected.sort()
    chosen = set((u, v) for _, u, v in undirected[:n])

    pred = np.zeros(e, dtype=np.float32)
    for k in range(e):
        u = int(edge_index[0, k])
        v = int(edge_index[1, k])
        a, b = min(u, v), max(u, v)
        if (a, b) in chosen:
            pred[k] = 1.0

    return pred
