import numpy as np
from data.held_karp import held_karp


Point = tuple[float, float]

class TSPGraphData:
    def __init__(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        y: np.ndarray,
    ):
        self.x = x                    # [n, 2]
        self.edge_index = edge_index  # [2, E]
        self.edge_attr = edge_attr    # [E, 3]
        self.y = y                    # [E]

    @staticmethod
    def from_points(points: list[Point]) -> "TSPGraphData":
        n = len(points)
        e = n * (n - 1)

        ans, path, tour_sum, isOpt, dist = held_karp(points)

        dist = np.array(dist, dtype=float)       # [n, n]
        isOpt = np.array(isOpt, dtype=bool)      # [n, n]
        path = np.array(path, dtype=int)         # [n]

        x = np.array(points, dtype=float)        # [n, 2]

        edge_index = np.zeros((2, e), dtype=int)
        edge_attr = np.zeros((e, 3), dtype=float)
        y = np.zeros(e, dtype=np.float32)

        ptr = 0
        for i in range(n):
            xi, yi = points[i]
            for j in range(n):
                if i == j:
                    continue

                xj, yj = points[j]

                edge_index[0, ptr] = i
                edge_index[1, ptr] = j

                dx = xj - xi
                dy = yj - yi
                d = dist[i, j]

                edge_attr[ptr] = [dx, dy, d]
                y[ptr] = float(isOpt[i, j])

                ptr += 1

        return TSPGraphData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )
