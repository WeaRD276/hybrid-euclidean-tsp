# Hybridization mechanisms, n = 20 (1000 test instances)

Mean test gap % over LKH; neural columns are the mean ± std over seeds. k = candidate edges per node taken from the heatmap. “vs SoftDist” sums per-instance wins / ties / losses of the neural runs against SoftDist with the same mechanism and k, and gives the largest Wilcoxon p over seeds.

Reference: greedy edge 12.66%, greedy edge + full 2-opt 1.22%

| mechanism | k | distance | kNN rank | SoftDist | graphconv, complete | vs SoftDist (W / T / L, max p) |
| --- | --- | --- | --- | --- | --- | --- |
| fused greedy |  |  | 10.65 | 9.67 | 6.68 ± 0.27 | 1851 / 302 / 847, p < 1e-4 |
| pruned greedy | 2 |  | 10.67 | 9.73 | 6.82 ± 0.29 | 1833 / 306 / 861, p < 1e-4 |
| pruned greedy | 3 |  | 10.69 | 9.54 | 6.75 ± 0.31 | 1816 / 317 / 867, p < 1e-4 |
| pruned greedy | 5 |  | 10.61 | 9.65 | 6.68 ± 0.27 | 1855 / 295 / 850, p < 1e-4 |
| fused + full 2-opt |  |  | 1.08 | 0.89 | 0.81 ± 0.05 | 688 / 1684 / 628, p ≤ 0.62 |
| greedy + 2-opt cand | 2 | 4.92 | 5.23 | 5.07 | 4.41 ± 0.11 | 752 / 1728 / 520, p ≤ 0.0001 |
| greedy + 2-opt cand | 3 | 2.31 | 2.49 | 2.43 | 2.11 ± 0.06 | 450 / 2196 / 354, p ≤ 0.0089 |
| greedy + 2-opt cand | 5 | 1.30 | 1.32 | 1.28 | 1.29 ± 0.02 | 88 / 2803 / 109, p ≤ 0.89 |
| greedy + 2-opt cand | 8 | 1.22 | 1.22 | 1.22 | 1.23 ± 0.00 | 0 / 2993 / 7, p ≤ 0.18 |
| fused + 2-opt cand | 2 |  | 5.41 | 5.21 | 3.88 ± 0.30 | 1344 / 796 / 860, p < 1e-4 |
| fused + 2-opt cand | 3 |  | 2.48 | 2.19 | 1.84 ± 0.15 | 1003 / 1184 / 813, p ≤ 0.21 |
| fused + 2-opt cand | 5 |  | 1.16 | 0.95 | 0.94 ± 0.06 | 707 / 1607 / 686, p ≤ 0.56 |
| fused + 2-opt cand | 8 |  | 1.08 | 0.89 | 0.82 ± 0.04 | 686 / 1680 / 634, p ≤ 0.61 |
