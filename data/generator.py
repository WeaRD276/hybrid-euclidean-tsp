import random
import numpy as np

LOW_N = 3
HIGH_N = 12
SEED = 47

def gen_points(seed, N = -1):
    # rng = np.random.default_rng(seed)
    rng = random.Random(seed)
    if N == -1:
        N = rng.randint(LOW_N, HIGH_N)

    points = []
    for i in range(N):
        # x = rng.random()
        # y = rng.random()

        x = rng.uniform(0, 1)
        y = rng.uniform(0, 1)

        # x = rng.random()
        # y = rng.random()

        points.append([x, y])
    return points
