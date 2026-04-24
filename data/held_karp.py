import math
import sys

INF = float('inf')
EPS = 1e-9

def held_karp(points):
    n = len(points)

    if n < 3:
        raise ValueError("should have at least 3 points")

    dist = [[0.0] * n for _ in range(n)]
    isOpt = [[0] * n for _ in range(n)]

    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            d = math.hypot(xi - xj, yi - yj)
            dist[i][j] = dist[j][i] = d

    size = 1 << n
    dp = [[-1.0] * n for _ in range(size)]
    parent = [[(-1, -1)] * n for _ in range(size)]

    # sys.setrecursionlimit(1 << 25)

    def f(mask, pos):
        if mask.bit_count() == n:
            return dist[pos][0]
        if dp[mask][pos] != -1.0:
            return dp[mask][pos]

        res = INF
        best_next = (-1, -1)

        for nxt in range(n):
            if mask & (1 << nxt):
                continue

            cur = dist[pos][nxt] + f(mask | (1 << nxt), nxt)
            if cur < res:
                res = cur
                best_next = (mask | (1 << nxt), nxt)

        dp[mask][pos] = res
        parent[mask][pos] = best_next
        return res

    ans = f(1, 0)

    path = []
    mask, v = 1, 0
    while v != -1:
        path.append(v)
        mask, v = parent[mask][v]

    tour_sum = 0.0
    for i in range(n):
        a = path[i]
        b = path[(i + 1) % n]
        isOpt[a][b] = isOpt[b][a] = 1
        tour_sum += dist[a][b]

    assert abs(ans - tour_sum) < EPS

    return ans, path, tour_sum, isOpt, dist


def solve():
    data = sys.stdin.read().strip().split()
    if not data:
        return

    it = iter(data)
    n = int(next(it))
    points = []
    for _ in range(n):
        x = float(next(it))
        y = float(next(it))
        points.append((x, y))

    ans, path, tour_sum, isOpt, dist = held_karp(points)

    print(ans)
    print(*path)
    print(tour_sum)


if __name__ == "__main__":
    solve()
