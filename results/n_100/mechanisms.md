# Hybridization mechanisms, n = 100 (1000 test instances)

Mean test gap % over LKH; neural columns are the mean ± std over seeds. k = candidate edges per node taken from the heatmap. “vs SoftDist” sums per-instance wins / ties / losses of the neural runs against SoftDist with the same mechanism and k, and gives the largest Wilcoxon p over seeds.

Reference: greedy edge 18.03%, greedy edge + full 2-opt 3.53%

| mechanism | k | distance | kNN rank | SoftDist | graphconv, kNN k=10 | vs SoftDist (W / T / L, max p) |
| --- | --- | --- | --- | --- | --- | --- |
| fused greedy |  |  | 16.12 | 16.01 | 14.83 ± 0.12 | 1740 / 0 / 1260, p < 1e-4 |
| pruned greedy | 2 |  | 16.07 | 16.14 | 15.00 ± 0.07 | 1712 / 0 / 1288, p < 1e-4 |
| pruned greedy | 3 |  | 16.12 | 16.00 | 14.93 ± 0.12 | 1733 / 0 / 1267, p < 1e-4 |
| pruned greedy | 5 |  | 16.11 | 15.88 | 14.87 ± 0.13 | 1721 / 0 / 1279, p < 1e-4 |
| fused + full 2-opt |  |  | 3.17 | 2.99 | 2.88 ± 0.02 | 1490 / 31 / 1479, p ≤ 0.63 |
| greedy + 2-opt cand | 2 | 12.63 | 12.90 | 12.69 | 12.47 ± 0.03 | 1452 / 494 / 1054, p ≤ 0.00015 |
| greedy + 2-opt cand | 3 | 9.00 | 9.26 | 9.02 | 8.86 ± 0.01 | 1300 / 542 / 1158, p ≤ 0.046 |
| greedy + 2-opt cand | 5 | 5.12 | 5.28 | 5.13 | 5.05 ± 0.06 | 957 / 1169 / 874, p ≤ 0.68 |
| greedy + 2-opt cand | 8 | 3.93 | 4.05 | 3.91 | 3.94 ± 0.02 | 430 / 2075 / 495, p ≤ 0.48 |
| fused + 2-opt cand | 2 |  | 12.40 | 12.54 | 11.52 ± 0.09 | 1716 / 2 / 1282, p < 1e-4 |
| fused + 2-opt cand | 3 |  | 8.99 | 9.04 | 8.39 ± 0.07 | 1665 / 8 / 1327, p ≤ 0.00075 |
| fused + 2-opt cand | 5 |  | 4.98 | 4.72 | 4.57 ± 0.02 | 1519 / 12 / 1469, p ≤ 0.47 |
| fused + 2-opt cand | 8 |  | 3.61 | 3.37 | 3.40 ± 0.01 | 1473 / 26 / 1501, p ≤ 0.79 |
