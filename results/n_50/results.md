# Tour quality, n = 50 (1000 test instances)

Gap is relative to the LKH reference tour (optimal at this size in practice). λ and the score mode are chosen on the validation split. W/T/L and the Wilcoxon p-value compare each method per instance against the row named in “vs”.

| method | setting | mean gap % [95% CI] | ms / instance | vs | W / T / L | p |
| --- | --- | --- | ---: | --- | --- | ---: |
| greedy edge |  | 17.02 [16.62, 17.43] | 0.42 |  |  |  |
| nearest neighbour |  | 22.66 [22.15, 23.17] | 0.09 | greedy edge | 274 / 0 / 726 | <1e-4 |
| farthest insertion |  | 5.76 [5.57, 5.96] | 1.11 | greedy edge | 950 / 0 / 50 | <1e-4 |
| greedy edge + 2-opt |  | 2.92 [2.77, 3.07] | 0.56 | greedy edge | 1000 / 0 / 0 | <1e-4 |
| farthest insertion + 2-opt |  | 4.63 [4.46, 4.80] | 0.17 | greedy edge + 2-opt | 295 / 10 / 695 | <1e-4 |
| greedy + kNN rank (no learning) | raw, λ=1000 | 14.88 [14.51, 15.24] | 0.46 | greedy edge | 572 / 35 / 393 | <1e-4 |
| greedy + kNN rank (no learning) + 2-opt | raw, λ=1000 | 2.58 [2.45, 2.71] | 0.89 | greedy edge + 2-opt | 448 / 161 / 391 | <1e-4 |
| greedy + SoftDist (no learning) | raw, λ=1000 | 14.09 [13.72, 14.46] | 0.45 | greedy edge | 631 / 27 / 342 | <1e-4 |
| greedy + SoftDist (no learning) + 2-opt | raw, λ=1000 | 2.37 [2.25, 2.50] | 0.86 | greedy edge + 2-opt | 501 / 166 / 333 | <1e-4 |
| greedy + n50_graphconv_complete_h64_s0 | raw, λ=1000 | 16.89 [16.48, 17.31] | 0.54 | greedy edge | 134 / 763 / 103 | 0.0783 |
| greedy + n50_graphconv_complete_h64_s0 + 2-opt | raw, λ=1000 | 2.86 [2.72, 2.99] | 1.00 | greedy edge + 2-opt | 96 / 849 / 55 | 0.0074 |
| greedy + n50_graphconv_complete_h64_s1 | raw, λ=1000 | 11.18 [10.86, 11.49] | 0.42 | greedy edge | 797 / 5 / 198 | <1e-4 |
| greedy + n50_graphconv_complete_h64_s1 + 2-opt | raw, λ=1000 | 2.19 [2.07, 2.30] | 0.80 | greedy edge + 2-opt | 534 / 141 / 325 | <1e-4 |
| greedy + n50_graphconv_complete_h64_s2 | raw, λ=1000 | 12.51 [12.17, 12.85] | 0.44 | greedy edge | 720 / 35 / 245 | <1e-4 |
| greedy + n50_graphconv_complete_h64_s2 + 2-opt | raw, λ=1000 | 2.30 [2.18, 2.42] | 0.83 | greedy edge + 2-opt | 462 / 249 / 289 | <1e-4 |
| greedy + n50_graphconv_knn10_h64_s0 | raw, λ=5 | 12.61 [12.27, 12.97] | 0.39 | greedy edge | 696 / 0 / 304 | <1e-4 |
| greedy + n50_graphconv_knn10_h64_s0 + 2-opt | raw, λ=5 | 2.28 [2.16, 2.40] | 0.79 | greedy edge + 2-opt | 552 / 59 / 389 | <1e-4 |
| greedy + n50_graphconv_knn10_h64_s1 | raw, λ=10 | 12.58 [12.24, 12.94] | 0.43 | greedy edge | 714 / 0 / 286 | <1e-4 |
| greedy + n50_graphconv_knn10_h64_s1 + 2-opt | raw, λ=10 | 2.34 [2.22, 2.46] | 0.82 | greedy edge + 2-opt | 539 / 71 / 390 | <1e-4 |
| greedy + n50_graphconv_knn10_h64_s2 | norm, λ=16 | 12.98 [12.64, 13.33] | 0.51 | greedy edge | 694 / 1 / 305 | <1e-4 |
| greedy + n50_graphconv_knn10_h64_s2 + 2-opt | norm, λ=16 | 2.33 [2.20, 2.45] | 0.92 | greedy edge + 2-opt | 540 / 70 / 390 | <1e-4 |

## Neural runs aggregated over seeds

| configuration | variant | seeds | mean gap % (mean ± std over seeds) |
| --- | --- | ---: | --- |
| graphconv, complete | greedy | 3 | 13.53 ± 2.44 |
| graphconv, complete | + 2-opt | 3 | 2.45 ± 0.29 |
| graphconv, kNN k=10 | greedy | 3 | 12.72 ± 0.18 |
| graphconv, kNN k=10 | + 2-opt | 3 | 2.31 ± 0.03 |

# Edge prediction, n = 50

Computed over undirected pairs; p(u,v) and p(v,u) are averaged. AP = average precision. P@n = share of tour edges among the n top-scoring pairs. R@k = share of each node's two tour neighbours inside its top-k. Length ratio = mean length of the n top pairs ÷ mean tour-edge length (below 1 = biased to short edges). F1 uses the threshold that maximises F1 on validation (neural runs only).

| heatmap | AP | P@n | R@2 | R@5 | length ratio | thr | P | R | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| distance only (n shortest edges) | 0.655 | 0.609 | 0.682 | 0.930 | 0.702 |  |  |  |  |
| kNN rank (no learning) | 0.790 | 0.711 | 0.718 | 0.956 | 0.801 |  |  |  |  |
| SoftDist (no learning) | 0.770 | 0.696 | 0.714 | 0.953 | 0.781 |  |  |  |  |
| n50_graphconv_complete_h64_s0 | 0.656 | 0.612 | 0.683 | 0.931 | 0.702 | 0.80 | 0.538 | 0.699 | 0.608 |
| n50_graphconv_complete_h64_s1 | 0.665 | 0.633 | 0.693 | 0.943 | 0.795 | 0.90 | 0.565 | 0.720 | 0.633 |
| n50_graphconv_complete_h64_s2 | 0.660 | 0.623 | 0.693 | 0.941 | 0.741 | 0.90 | 0.551 | 0.719 | 0.624 |
| n50_graphconv_knn10_h64_s0 | 0.760 | 0.689 | 0.713 | 0.960 | 0.837 | 0.75 | 0.667 | 0.716 | 0.690 |
| n50_graphconv_knn10_h64_s1 | 0.762 | 0.692 | 0.713 | 0.961 | 0.851 | 0.75 | 0.653 | 0.740 | 0.693 |
| n50_graphconv_knn10_h64_s2 | 0.758 | 0.689 | 0.712 | 0.960 | 0.844 | 0.75 | 0.665 | 0.717 | 0.690 |
