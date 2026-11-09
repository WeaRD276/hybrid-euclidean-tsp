# Tour quality, n = 100 (1000 test instances)

Gap is relative to the LKH reference tour (optimal at this size in practice). λ and the score mode are chosen on the validation split. W/T/L and the Wilcoxon p-value compare each method per instance against the row named in “vs”.

| method | setting | mean gap % [95% CI] | ms / instance | vs | W / T / L | p |
| --- | --- | --- | ---: | --- | --- | ---: |
| greedy edge |  | 18.03 [17.69, 18.37] | 1.81 |  |  |  |
| nearest neighbour |  | 24.95 [24.55, 25.33] | 0.20 | greedy edge | 179 / 0 / 821 | <1e-4 |
| farthest insertion |  | 7.50 [7.35, 7.65] | 1.87 | greedy edge | 971 / 0 / 29 | <1e-4 |
| greedy edge + 2-opt |  | 3.53 [3.41, 3.64] | 2.23 | greedy edge | 1000 / 0 / 0 | <1e-4 |
| farthest insertion + 2-opt |  | 6.34 [6.20, 6.49] | 0.55 | greedy edge + 2-opt | 161 / 0 / 839 | <1e-4 |
| greedy + kNN rank (no learning) | raw, λ=1000 | 16.12 [15.82, 16.41] | 1.64 | greedy edge | 626 / 1 / 373 | <1e-4 |
| greedy + kNN rank (no learning) + 2-opt | raw, λ=1000 | 3.17 [3.07, 3.27] | 3.38 | greedy edge + 2-opt | 541 / 10 / 449 | <1e-4 |
| greedy + SoftDist (no learning) | raw, λ=1000 | 16.01 [15.71, 16.30] | 1.81 | greedy edge | 620 / 1 / 379 | <1e-4 |
| greedy + SoftDist (no learning) + 2-opt | raw, λ=1000 | 2.99 [2.88, 3.09] | 3.53 | greedy edge + 2-opt | 598 / 20 / 382 | <1e-4 |
| greedy + n100_graphconv_knn10_h64_s0 | raw, λ=50 | 14.67 [14.38, 14.95] | 1.61 | greedy edge | 702 / 0 / 298 | <1e-4 |
| greedy + n100_graphconv_knn10_h64_s0 + 2-opt | raw, λ=50 | 2.89 [2.80, 2.99] | 3.38 | greedy edge + 2-opt | 607 / 1 / 392 | <1e-4 |
| greedy + n100_graphconv_knn10_h64_s1 | raw, λ=50 | 14.86 [14.57, 15.15] | 1.68 | greedy edge | 681 / 0 / 319 | <1e-4 |
| greedy + n100_graphconv_knn10_h64_s1 + 2-opt | raw, λ=50 | 2.86 [2.76, 2.95] | 3.57 | greedy edge + 2-opt | 640 / 4 / 356 | <1e-4 |
| greedy + n100_graphconv_knn10_h64_s2 | raw, λ=1000 | 14.95 [14.65, 15.25] | 1.62 | greedy edge | 660 / 0 / 340 | <1e-4 |
| greedy + n100_graphconv_knn10_h64_s2 + 2-opt | raw, λ=1000 | 2.90 [2.80, 3.00] | 3.17 | greedy edge + 2-opt | 614 / 5 / 381 | <1e-4 |

## Neural runs aggregated over seeds

| configuration | variant | seeds | mean gap % (mean ± std over seeds) |
| --- | --- | ---: | --- |
| graphconv, kNN k=10 | greedy | 3 | 14.83 ± 0.12 |
| graphconv, kNN k=10 | + 2-opt | 3 | 2.88 ± 0.02 |

# Edge prediction, n = 100

Computed over undirected pairs; p(u,v) and p(v,u) are averaged. AP = average precision. P@n = share of tour edges among the n top-scoring pairs. R@k = share of each node's two tour neighbours inside its top-k. Length ratio = mean length of the n top pairs ÷ mean tour-edge length (below 1 = biased to short edges). F1 uses the threshold that maximises F1 on validation (neural runs only).

| heatmap | AP | P@n | R@2 | R@5 | length ratio | thr | P | R | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| distance only (n shortest edges) | 0.658 | 0.611 | 0.688 | 0.935 | 0.708 |  |  |  |  |
| kNN rank (no learning) | 0.797 | 0.717 | 0.724 | 0.958 | 0.812 |  |  |  |  |
| SoftDist (no learning) | 0.774 | 0.697 | 0.718 | 0.956 | 0.790 |  |  |  |  |
| n100_graphconv_knn10_h64_s0 | 0.742 | 0.678 | 0.708 | 0.958 | 0.822 | 0.75 | 0.635 | 0.736 | 0.681 |
| n100_graphconv_knn10_h64_s1 | 0.744 | 0.679 | 0.709 | 0.959 | 0.825 | 0.70 | 0.630 | 0.743 | 0.682 |
| n100_graphconv_knn10_h64_s2 | 0.742 | 0.677 | 0.708 | 0.959 | 0.824 | 0.65 | 0.626 | 0.746 | 0.681 |
