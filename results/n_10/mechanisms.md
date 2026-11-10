# Hybridization mechanisms, n = 10 (1000 test instances)

Mean test gap % over LKH; neural columns are the mean ± std over seeds. k = candidate edges per node taken from the heatmap. “vs SoftDist” sums per-instance wins / ties / losses of the neural runs against SoftDist with the same mechanism and k, and gives the largest Wilcoxon p over seeds.

Reference: greedy edge 8.06%, greedy edge + full 2-opt 0.28%

| mechanism | k | distance | kNN rank | SoftDist | graphconv, complete | vs SoftDist (W / T / L, max p) |
| --- | --- | --- | --- | --- | --- | --- |
| fused greedy |  |  | 7.06 | 5.59 | 1.74 ± 0.10 | 1783 / 762 / 455, p < 1e-4 |
| pruned greedy | 2 |  | 6.97 | 5.61 | 1.73 ± 0.10 | 1781 / 761 / 458, p < 1e-4 |
| pruned greedy | 3 |  | 7.01 | 5.60 | 1.73 ± 0.10 | 1797 / 758 / 445, p < 1e-4 |
| pruned greedy | 5 |  | 7.04 | 5.55 | 1.74 ± 0.10 | 1782 / 763 / 455, p < 1e-4 |
| fused + full 2-opt |  |  | 0.22 | 0.19 | 0.17 ± 0.00 | 224 / 2558 / 218, p ≤ 0.93 |
| greedy + 2-opt cand | 2 | 1.39 | 1.51 | 1.41 | 0.59 ± 0.02 | 518 / 2347 / 135, p < 1e-4 |
| greedy + 2-opt cand | 3 | 0.45 | 0.53 | 0.49 | 0.35 ± 0.00 | 119 / 2828 / 53, p ≤ 0.00027 |
| greedy + 2-opt cand | 5 | 0.28 | 0.29 | 0.28 | 0.28 ± 0.00 | 0 / 3000 / 0, p ≤ 1 |
| greedy + 2-opt cand | 8 | 0.28 | 0.28 | 0.28 | 0.28 ± 0.00 | 0 / 3000 / 0, p ≤ 1 |
| fused + 2-opt cand | 2 |  | 1.83 | 1.55 | 0.46 ± 0.02 | 813 / 1852 / 335, p < 1e-4 |
| fused + 2-opt cand | 3 |  | 0.51 | 0.40 | 0.21 ± 0.00 | 345 / 2400 / 255, p ≤ 0.0034 |
| fused + 2-opt cand | 5 |  | 0.22 | 0.19 | 0.17 ± 0.00 | 224 / 2558 / 218, p ≤ 0.93 |
| fused + 2-opt cand | 8 |  | 0.22 | 0.19 | 0.17 ± 0.00 | 224 / 2558 / 218, p ≤ 0.93 |
