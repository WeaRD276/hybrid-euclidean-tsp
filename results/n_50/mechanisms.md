# Hybridization mechanisms, n = 50 (1000 test instances)

Mean test gap % over LKH; neural columns are the mean ± std over seeds. k = candidate edges per node taken from the heatmap. “vs SoftDist” sums per-instance wins / ties / losses of the neural runs against SoftDist with the same mechanism and k, and gives the largest Wilcoxon p over seeds.

Reference: greedy edge 17.02%, greedy edge + full 2-opt 2.92%

| mechanism | k | distance | kNN rank | SoftDist | graphconv, kNN k=10 | vs SoftDist (W / T / L, max p) |
| --- | --- | --- | --- | --- | --- | --- |
| fused greedy |  |  | 14.88 | 14.09 | 12.72 ± 0.18 | 1740 / 12 / 1248, p < 1e-4 |
| pruned greedy | 2 |  | 14.95 | 14.28 | 13.00 ± 0.09 | 1724 / 5 / 1271, p < 1e-4 |
| pruned greedy | 3 |  | 14.84 | 14.04 | 12.78 ± 0.20 | 1728 / 10 / 1262, p < 1e-4 |
| pruned greedy | 5 |  | 14.85 | 13.86 | 12.69 ± 0.19 | 1723 / 13 / 1264, p < 1e-4 |
| fused + full 2-opt |  |  | 2.58 | 2.37 | 2.31 ± 0.03 | 1349 / 323 / 1328, p ≤ 0.95 |
| greedy + 2-opt cand | 2 | 10.31 | 10.68 | 10.48 | 9.79 ± 0.03 | 1343 / 757 / 900, p < 1e-4 |
| greedy + 2-opt cand | 3 | 6.52 | 6.95 | 6.67 | 6.30 ± 0.04 | 1155 / 906 / 939, p ≤ 0.0011 |
| greedy + 2-opt cand | 5 | 3.66 | 3.72 | 3.61 | 3.63 ± 0.01 | 641 / 1718 / 641, p ≤ 0.95 |
| greedy + 2-opt cand | 8 | 3.13 | 3.14 | 3.10 | 3.18 ± 0.01 | 156 / 2593 / 251, p ≤ 0.03 |
| fused + 2-opt cand | 2 |  | 10.39 | 10.14 | 9.32 ± 0.01 | 1658 / 47 / 1295, p ≤ 0.00013 |
| fused + 2-opt cand | 3 |  | 6.71 | 6.45 | 5.94 ± 0.05 | 1580 / 99 / 1321, p ≤ 0.0083 |
| fused + 2-opt cand | 5 |  | 3.60 | 3.22 | 3.24 ± 0.07 | 1349 / 194 / 1457, p ≤ 0.95 |
| fused + 2-opt cand | 8 |  | 2.82 | 2.60 | 2.57 ± 0.03 | 1363 / 284 / 1353, p ≤ 0.87 |
