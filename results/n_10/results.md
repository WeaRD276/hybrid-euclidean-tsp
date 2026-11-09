# Tour quality, n = 10 (1000 test instances)

Gap is relative to the LKH reference tour (optimal at this size in practice). λ and the score mode are chosen on the validation split. W/T/L and the Wilcoxon p-value compare each method per instance against the row named in “vs”.

| method | setting | mean gap % [95% CI] | ms / instance | vs | W / T / L | p |
| --- | --- | --- | ---: | --- | --- | ---: |
| greedy edge |  | 8.06 [7.60, 8.54] | 0.05 |  |  |  |
| nearest neighbour |  | 10.35 [9.82, 10.88] | 0.02 | greedy edge | 305 / 193 / 502 | <1e-4 |
| farthest insertion |  | 0.67 [0.57, 0.78] | 0.16 | greedy edge | 786 / 172 / 42 | <1e-4 |
| greedy edge + 2-opt |  | 0.28 [0.22, 0.34] | 0.06 | greedy edge | 785 / 215 / 0 | <1e-4 |
| farthest insertion + 2-opt |  | 0.35 [0.28, 0.41] | 0.03 | greedy edge + 2-opt | 87 / 787 / 126 | 0.0057 |
| greedy + kNN rank (no learning) | raw, λ=5 | 7.06 [6.62, 7.51] | 0.05 | greedy edge | 222 / 615 / 163 | <1e-4 |
| greedy + kNN rank (no learning) + 2-opt | raw, λ=5 | 0.22 [0.17, 0.28] | 0.11 | greedy edge + 2-opt | 45 / 927 / 28 | 0.0172 |
| greedy + SoftDist (no learning) | raw, λ=50 | 5.59 [5.20, 5.99] | 0.04 | greedy edge | 327 / 540 / 133 | <1e-4 |
| greedy + SoftDist (no learning) + 2-opt | raw, λ=50 | 0.19 [0.14, 0.24] | 0.10 | greedy edge + 2-opt | 51 / 918 / 31 | 0.0058 |
| greedy + n10_gat_complete_h64_s0 | norm, λ=4 | 7.54 [7.08, 8.00] | 0.07 | greedy edge | 117 / 795 / 88 | 0.0030 |
| greedy + n10_gat_complete_h64_s0 + 2-opt | norm, λ=4 | 0.29 [0.23, 0.35] | 0.13 | greedy edge + 2-opt | 14 / 960 / 26 | 0.2694 |
| greedy + n10_gat_complete_h64_s1 | raw, λ=1000 | 4.09 [3.78, 4.43] | 0.05 | greedy edge | 520 / 314 / 166 | <1e-4 |
| greedy + n10_gat_complete_h64_s1 + 2-opt | raw, λ=1000 | 0.20 [0.15, 0.25] | 0.11 | greedy edge + 2-opt | 79 / 857 / 64 | 0.0852 |
| greedy + n10_gat_complete_h64_s2 | raw, λ=1000 | 2.19 [1.98, 2.42] | 0.04 | greedy edge | 645 / 216 / 139 | <1e-4 |
| greedy + n10_gat_complete_h64_s2 + 2-opt | raw, λ=1000 | 0.17 [0.12, 0.21] | 0.09 | greedy edge + 2-opt | 97 / 842 / 61 | 0.0017 |
| greedy + n10_graphconv_complete_h64_s0 | raw, λ=2 | 1.70 [1.51, 1.90] | 0.04 | greedy edge | 694 / 178 / 128 | <1e-4 |
| greedy + n10_graphconv_complete_h64_s0 + 2-opt | raw, λ=2 | 0.17 [0.13, 0.21] | 0.09 | greedy edge + 2-opt | 95 / 830 / 75 | 0.0255 |
| greedy + n10_graphconv_complete_h64_s1 | raw, λ=5 | 1.88 [1.63, 2.13] | 0.04 | greedy edge | 703 / 172 / 125 | <1e-4 |
| greedy + n10_graphconv_complete_h64_s1 + 2-opt | raw, λ=5 | 0.18 [0.14, 0.23] | 0.08 | greedy edge + 2-opt | 102 / 816 / 82 | 0.0352 |
| greedy + n10_graphconv_complete_h64_s2 | norm, λ=16 | 1.65 [1.46, 1.85] | 0.07 | greedy edge | 684 / 184 / 132 | <1e-4 |
| greedy + n10_graphconv_complete_h64_s2 + 2-opt | norm, λ=16 | 0.18 [0.13, 0.22] | 0.10 | greedy edge + 2-opt | 95 / 838 / 67 | 0.0119 |
| greedy + n10_graphconv_knn5_h64_s0 | raw, λ=5 | 2.17 [1.91, 2.41] | 0.05 | greedy edge | 668 / 183 / 149 | <1e-4 |
| greedy + n10_graphconv_knn5_h64_s0 + 2-opt | raw, λ=5 | 0.16 [0.12, 0.21] | 0.09 | greedy edge + 2-opt | 95 / 843 / 62 | 0.0017 |
| greedy + n10_graphconv_knn5_h64_s1 | norm, λ=16 | 2.13 [1.87, 2.41] | 0.06 | greedy edge | 647 / 202 / 151 | <1e-4 |
| greedy + n10_graphconv_knn5_h64_s1 + 2-opt | norm, λ=16 | 0.14 [0.10, 0.18] | 0.13 | greedy edge + 2-opt | 95 / 847 / 58 | 0.0004 |
| greedy + n10_graphconv_knn5_h64_s2 | raw, λ=5 | 1.97 [1.76, 2.19] | 0.04 | greedy edge | 666 / 180 / 154 | <1e-4 |
| greedy + n10_graphconv_knn5_h64_s2 + 2-opt | raw, λ=5 | 0.13 [0.10, 0.17] | 0.07 | greedy edge + 2-opt | 96 / 847 / 57 | <1e-4 |

## Neural runs aggregated over seeds

| configuration | variant | seeds | mean gap % (mean ± std over seeds) |
| --- | --- | ---: | --- |
| gat, complete | greedy | 3 | 4.61 ± 2.21 |
| gat, complete | + 2-opt | 3 | 0.22 ± 0.05 |
| graphconv, complete | greedy | 3 | 1.74 ± 0.10 |
| graphconv, complete | + 2-opt | 3 | 0.17 ± 0.00 |
| graphconv, kNN k=5 | greedy | 3 | 2.09 ± 0.09 |
| graphconv, kNN k=5 | + 2-opt | 3 | 0.14 ± 0.01 |

# Edge prediction, n = 10

Computed over undirected pairs; p(u,v) and p(v,u) are averaged. AP = average precision. P@n = share of tour edges among the n top-scoring pairs. R@k = share of each node's two tour neighbours inside its top-k. Length ratio = mean length of the n top pairs ÷ mean tour-edge length (below 1 = biased to short edges). F1 uses the threshold that maximises F1 on validation (neural runs only).

| heatmap | AP | P@n | R@2 | R@5 | length ratio | thr | P | R | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| distance only (n shortest edges) | 0.665 | 0.626 | 0.675 | 0.935 | 0.711 |  |  |  |  |
| kNN rank (no learning) | 0.783 | 0.710 | 0.721 | 0.966 | 0.777 |  |  |  |  |
| SoftDist (no learning) | 0.783 | 0.707 | 0.722 | 0.964 | 0.753 |  |  |  |  |
| n10_gat_complete_h64_s0 | 0.679 | 0.629 | 0.681 | 0.942 | 0.718 | 0.65 | 0.575 | 0.688 | 0.627 |
| n10_gat_complete_h64_s1 | 0.759 | 0.693 | 0.722 | 0.971 | 0.795 | 0.65 | 0.644 | 0.744 | 0.690 |
| n10_gat_complete_h64_s2 | 0.794 | 0.728 | 0.743 | 0.982 | 0.851 | 0.65 | 0.666 | 0.792 | 0.723 |
| n10_graphconv_complete_h64_s0 | 0.907 | 0.820 | 0.835 | 0.995 | 0.947 | 0.70 | 0.791 | 0.849 | 0.819 |
| n10_graphconv_complete_h64_s1 | 0.907 | 0.818 | 0.839 | 0.995 | 0.946 | 0.70 | 0.792 | 0.851 | 0.820 |
| n10_graphconv_complete_h64_s2 | 0.896 | 0.809 | 0.825 | 0.994 | 0.923 | 0.65 | 0.776 | 0.848 | 0.810 |
| n10_graphconv_knn5_h64_s0 | 0.870 | 0.787 | 0.800 | 0.980 | 0.906 | 0.50 | 0.747 | 0.828 | 0.785 |
| n10_graphconv_knn5_h64_s1 | 0.887 | 0.802 | 0.820 | 0.982 | 0.919 | 0.60 | 0.788 | 0.819 | 0.804 |
| n10_graphconv_knn5_h64_s2 | 0.873 | 0.793 | 0.806 | 0.982 | 0.903 | 0.60 | 0.769 | 0.814 | 0.791 |
