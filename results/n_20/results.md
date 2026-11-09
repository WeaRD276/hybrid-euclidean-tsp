# Tour quality, n = 20 (1000 test instances)

Gap is relative to the LKH reference tour (optimal at this size in practice). λ and the score mode are chosen on the validation split. W/T/L and the Wilcoxon p-value compare each method per instance against the row named in “vs”.

| method | setting | mean gap % [95% CI] | ms / instance | vs | W / T / L | p |
| --- | --- | --- | ---: | --- | --- | ---: |
| greedy edge |  | 12.66 [12.17, 13.19] | 0.10 |  |  |  |
| nearest neighbour |  | 17.15 [16.53, 17.77] | 0.04 | greedy edge | 339 / 21 / 640 | <1e-4 |
| farthest insertion |  | 2.51 [2.32, 2.71] | 0.30 | greedy edge | 896 / 15 / 89 | <1e-4 |
| greedy edge + 2-opt |  | 1.22 [1.09, 1.35] | 0.13 | greedy edge | 966 / 34 / 0 | <1e-4 |
| farthest insertion + 2-opt |  | 1.55 [1.40, 1.69] | 0.06 | greedy edge + 2-opt | 273 / 359 / 368 | <1e-4 |
| greedy + kNN rank (no learning) | raw, λ=1000 | 10.65 [10.22, 11.11] | 0.09 | greedy edge | 447 / 274 / 279 | <1e-4 |
| greedy + kNN rank (no learning) + 2-opt | raw, λ=1000 | 1.08 [0.96, 1.19] | 0.21 | greedy edge + 2-opt | 194 / 653 / 153 | 0.0188 |
| greedy + SoftDist (no learning) | raw, λ=1000 | 9.67 [9.23, 10.11] | 0.08 | greedy edge | 510 / 240 / 250 | <1e-4 |
| greedy + SoftDist (no learning) + 2-opt | raw, λ=1000 | 0.89 [0.79, 0.99] | 0.22 | greedy edge + 2-opt | 212 / 652 / 136 | <1e-4 |
| greedy + n20_gat_complete_h64_s0 | raw, λ=1000 | 12.26 [11.76, 12.77] | 0.08 | greedy edge | 222 / 594 / 184 | 0.0333 |
| greedy + n20_gat_complete_h64_s0 + 2-opt | raw, λ=1000 | 1.26 [1.12, 1.40] | 0.18 | greedy edge + 2-opt | 79 / 831 / 90 | 0.4850 |
| greedy + n20_gat_complete_h64_s1 | norm, λ=8 | 12.40 [11.89, 12.92] | 0.13 | greedy edge | 138 / 741 / 121 | 0.0618 |
| greedy + n20_gat_complete_h64_s1 + 2-opt | norm, λ=8 | 1.23 [1.10, 1.36] | 0.25 | greedy edge + 2-opt | 45 / 901 / 54 | 0.4979 |
| greedy + n20_gat_complete_h64_s2 | raw, λ=50 | 12.43 [11.93, 12.94] | 0.10 | greedy edge | 186 / 652 / 162 | 0.1512 |
| greedy + n20_gat_complete_h64_s2 + 2-opt | raw, λ=50 | 1.26 [1.12, 1.41] | 0.23 | greedy edge + 2-opt | 72 / 856 / 72 | 0.7381 |
| greedy + n20_gat_knn10_h64_s0 | raw, λ=5 | 7.97 [7.56, 8.40] | 0.09 | greedy edge | 661 / 72 / 267 | <1e-4 |
| greedy + n20_gat_knn10_h64_s0 + 2-opt | raw, λ=5 | 1.02 [0.91, 1.12] | 0.20 | greedy edge + 2-opt | 255 / 523 / 222 | 0.0229 |
| greedy + n20_gat_knn10_h64_s1 | norm, λ=128 | 7.80 [7.39, 8.25] | 0.14 | greedy edge | 679 / 55 / 266 | <1e-4 |
| greedy + n20_gat_knn10_h64_s1 + 2-opt | norm, λ=128 | 0.92 [0.82, 1.02] | 0.24 | greedy edge + 2-opt | 263 / 539 / 198 | <1e-4 |
| greedy + n20_gat_knn10_h64_s2 | raw, λ=1000 | 7.60 [7.21, 7.99] | 0.09 | greedy edge | 698 / 43 / 259 | <1e-4 |
| greedy + n20_gat_knn10_h64_s2 + 2-opt | raw, λ=1000 | 0.93 [0.83, 1.03] | 0.18 | greedy edge + 2-opt | 277 / 502 / 221 | 0.0005 |
| greedy + n20_gatv2e_complete_h64_s0 | norm, λ=32 | 7.38 [6.99, 7.78] | 0.10 | greedy edge | 695 / 49 / 256 | <1e-4 |
| greedy + n20_gatv2e_complete_h64_s0 + 2-opt | norm, λ=32 | 0.94 [0.83, 1.04] | 0.18 | greedy edge + 2-opt | 291 / 491 / 218 | 0.0001 |
| greedy + n20_gatv2e_complete_h64_s1 | raw, λ=1000 | 6.16 [5.81, 6.51] | 0.08 | greedy edge | 757 / 29 / 214 | <1e-4 |
| greedy + n20_gatv2e_complete_h64_s1 + 2-opt | raw, λ=1000 | 0.97 [0.87, 1.08] | 0.18 | greedy edge + 2-opt | 293 / 479 / 228 | 0.0011 |
| greedy + n20_gatv2e_complete_h64_s2 | raw, λ=5 | 6.07 [5.69, 6.46] | 0.08 | greedy edge | 760 / 35 / 205 | <1e-4 |
| greedy + n20_gatv2e_complete_h64_s2 + 2-opt | raw, λ=5 | 0.77 [0.68, 0.87] | 0.17 | greedy edge + 2-opt | 317 / 488 / 195 | <1e-4 |
| greedy + n20_graphconv_complete_h64_s0 | raw, λ=1000 | 6.37 [6.00, 6.75] | 0.09 | greedy edge | 760 / 46 / 194 | <1e-4 |
| greedy + n20_graphconv_complete_h64_s0 + 2-opt | raw, λ=1000 | 0.77 [0.68, 0.86] | 0.21 | greedy edge + 2-opt | 289 / 522 / 189 | <1e-4 |
| greedy + n20_graphconv_complete_h64_s1 | raw, λ=1000 | 7.02 [6.63, 7.43] | 0.09 | greedy edge | 730 / 46 / 224 | <1e-4 |
| greedy + n20_graphconv_complete_h64_s1 + 2-opt | raw, λ=1000 | 0.88 [0.78, 0.98] | 0.17 | greedy edge + 2-opt | 288 / 498 / 214 | <1e-4 |
| greedy + n20_graphconv_complete_h64_s2 | raw, λ=1000 | 6.66 [6.27, 7.05] | 0.09 | greedy edge | 738 / 38 / 224 | <1e-4 |
| greedy + n20_graphconv_complete_h64_s2 + 2-opt | raw, λ=1000 | 0.78 [0.70, 0.87] | 0.19 | greedy edge + 2-opt | 308 / 494 / 198 | <1e-4 |
| greedy + n20_graphconv_complete_h64_t500_e600_s0 | raw, λ=10 | 7.02 [6.65, 7.41] | 0.14 | greedy edge | 707 / 56 / 237 | <1e-4 |
| greedy + n20_graphconv_complete_h64_t500_e600_s0 + 2-opt | raw, λ=10 | 0.83 [0.74, 0.93] | 0.29 | greedy edge + 2-opt | 279 / 512 / 209 | <1e-4 |
| greedy + n20_graphconv_complete_h64_t500_e600_s1 | raw, λ=50 | 6.78 [6.41, 7.17] | 0.09 | greedy edge | 733 / 38 / 229 | <1e-4 |
| greedy + n20_graphconv_complete_h64_t500_e600_s1 + 2-opt | raw, λ=50 | 0.88 [0.78, 0.99] | 0.18 | greedy edge + 2-opt | 286 / 492 / 222 | <1e-4 |
| greedy + n20_graphconv_complete_h64_t500_e600_s2 | norm, λ=32 | 7.01 [6.64, 7.41] | 0.12 | greedy edge | 702 / 65 / 233 | <1e-4 |
| greedy + n20_graphconv_complete_h64_t500_e600_s2 + 2-opt | norm, λ=32 | 0.79 [0.71, 0.89] | 0.21 | greedy edge + 2-opt | 269 / 541 / 190 | <1e-4 |
| greedy + n20_graphconv_complete_h64_t500_s0 | raw, λ=10 | 12.65 [12.17, 13.16] | 0.10 | greedy edge | 40 / 912 / 48 | 0.8421 |
| greedy + n20_graphconv_complete_h64_t500_s0 + 2-opt | raw, λ=10 | 1.22 [1.09, 1.35] | 0.23 | greedy edge + 2-opt | 14 / 975 / 11 | 0.5439 |
| greedy + n20_graphconv_complete_h64_t500_s1 | raw, λ=1000 | 12.65 [12.16, 13.17] | 0.12 | greedy edge | 5 / 988 / 7 | 0.8753 |
| greedy + n20_graphconv_complete_h64_t500_s1 + 2-opt | raw, λ=1000 | 1.23 [1.09, 1.36] | 0.29 | greedy edge + 2-opt | 0 / 998 / 2 | 0.2850 |
| greedy + n20_graphconv_complete_h64_t500_s2 | norm, λ=128 | 12.61 [12.11, 13.12] | 0.13 | greedy edge | 17 / 969 / 14 | 0.4217 |
| greedy + n20_graphconv_complete_h64_t500_s2 + 2-opt | norm, λ=128 | 1.23 [1.10, 1.36] | 0.26 | greedy edge + 2-opt | 3 / 991 / 6 | 0.4496 |
| greedy + n20_graphconv_knn10_h64_s0 | raw, λ=10 | 6.66 [6.29, 7.05] | 0.19 | greedy edge | 751 / 24 / 225 | <1e-4 |
| greedy + n20_graphconv_knn10_h64_s0 + 2-opt | raw, λ=10 | 0.76 [0.67, 0.84] | 0.26 | greedy edge + 2-opt | 307 / 490 / 203 | <1e-4 |
| greedy + n20_graphconv_knn10_h64_s1 | raw, λ=10 | 6.95 [6.56, 7.37] | 0.09 | greedy edge | 743 / 17 / 240 | <1e-4 |
| greedy + n20_graphconv_knn10_h64_s1 + 2-opt | raw, λ=10 | 0.82 [0.71, 0.92] | 0.21 | greedy edge + 2-opt | 307 / 488 / 205 | <1e-4 |
| greedy + n20_graphconv_knn10_h64_s2 | norm, λ=8 | 6.85 [6.48, 7.24] | 0.09 | greedy edge | 719 / 58 / 223 | <1e-4 |
| greedy + n20_graphconv_knn10_h64_s2 + 2-opt | norm, λ=8 | 0.78 [0.69, 0.87] | 0.16 | greedy edge + 2-opt | 286 / 520 / 194 | <1e-4 |
| greedy + n20_graphconv_knn5_h64_s0 | raw, λ=1.5 | 7.20 [6.81, 7.61] | 0.11 | greedy edge | 710 / 35 / 255 | <1e-4 |
| greedy + n20_graphconv_knn5_h64_s0 + 2-opt | raw, λ=1.5 | 0.80 [0.70, 0.90] | 0.20 | greedy edge + 2-opt | 292 / 530 / 178 | <1e-4 |
| greedy + n20_graphconv_knn5_h64_s1 | raw, λ=1.5 | 7.43 [7.02, 7.85] | 0.10 | greedy edge | 689 / 47 / 264 | <1e-4 |
| greedy + n20_graphconv_knn5_h64_s1 + 2-opt | raw, λ=1.5 | 0.77 [0.68, 0.86] | 0.21 | greedy edge + 2-opt | 295 / 513 / 192 | <1e-4 |
| greedy + n20_graphconv_knn5_h64_s2 | raw, λ=5 | 7.41 [7.03, 7.86] | 0.07 | greedy edge | 707 / 26 / 267 | <1e-4 |
| greedy + n20_graphconv_knn5_h64_s2 + 2-opt | raw, λ=5 | 0.86 [0.75, 0.97] | 0.17 | greedy edge + 2-opt | 323 / 459 / 218 | <1e-4 |

## Neural runs aggregated over seeds

| configuration | variant | seeds | mean gap % (mean ± std over seeds) |
| --- | --- | ---: | --- |
| gat, complete | greedy | 3 | 12.36 ± 0.08 |
| gat, complete | + 2-opt | 3 | 1.25 ± 0.02 |
| gat, kNN k=10 | greedy | 3 | 7.79 ± 0.15 |
| gat, kNN k=10 | + 2-opt | 3 | 0.96 ± 0.04 |
| gatv2e, complete | greedy | 3 | 6.54 ± 0.60 |
| gatv2e, complete | + 2-opt | 3 | 0.89 ± 0.09 |
| graphconv, complete | greedy | 3 | 6.68 ± 0.27 |
| graphconv, complete | + 2-opt | 3 | 0.81 ± 0.05 |
| graphconv, complete, 500 train | greedy | 3 | 12.63 ± 0.02 |
| graphconv, complete, 500 train | + 2-opt | 3 | 1.23 ± 0.01 |
| graphconv, complete, 500 train, 600 epochs | greedy | 3 | 6.94 ± 0.11 |
| graphconv, complete, 500 train, 600 epochs | + 2-opt | 3 | 0.84 ± 0.04 |
| graphconv, kNN k=10 | greedy | 3 | 6.82 ± 0.12 |
| graphconv, kNN k=10 | + 2-opt | 3 | 0.78 ± 0.03 |
| graphconv, kNN k=5 | greedy | 3 | 7.35 ± 0.11 |
| graphconv, kNN k=5 | + 2-opt | 3 | 0.81 ± 0.04 |

# Edge prediction, n = 20

Computed over undirected pairs; p(u,v) and p(v,u) are averaged. AP = average precision. P@n = share of tour edges among the n top-scoring pairs. R@k = share of each node's two tour neighbours inside its top-k. Length ratio = mean length of the n top pairs ÷ mean tour-edge length (below 1 = biased to short edges). F1 uses the threshold that maximises F1 on validation (neural runs only).

| heatmap | AP | P@n | R@2 | R@5 | length ratio | thr | P | R | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| distance only (n shortest edges) | 0.651 | 0.613 | 0.679 | 0.928 | 0.693 |  |  |  |  |
| kNN rank (no learning) | 0.785 | 0.709 | 0.717 | 0.959 | 0.784 |  |  |  |  |
| SoftDist (no learning) | 0.775 | 0.702 | 0.715 | 0.954 | 0.757 |  |  |  |  |
| n20_gat_complete_h64_s0 | 0.655 | 0.613 | 0.680 | 0.928 | 0.694 | 0.80 | 0.561 | 0.669 | 0.610 |
| n20_gat_complete_h64_s1 | 0.655 | 0.613 | 0.680 | 0.928 | 0.694 | 0.80 | 0.558 | 0.676 | 0.611 |
| n20_gat_complete_h64_s2 | 0.655 | 0.612 | 0.679 | 0.928 | 0.694 | 0.80 | 0.557 | 0.675 | 0.610 |
| n20_gat_knn10_h64_s0 | 0.759 | 0.695 | 0.716 | 0.965 | 0.819 | 0.75 | 0.641 | 0.755 | 0.693 |
| n20_gat_knn10_h64_s1 | 0.759 | 0.691 | 0.719 | 0.965 | 0.814 | 0.75 | 0.665 | 0.722 | 0.693 |
| n20_gat_knn10_h64_s2 | 0.761 | 0.692 | 0.718 | 0.965 | 0.819 | 0.80 | 0.669 | 0.716 | 0.692 |
| n20_gatv2e_complete_h64_s0 | 0.818 | 0.743 | 0.760 | 0.978 | 0.898 | 0.85 | 0.724 | 0.763 | 0.743 |
| n20_gatv2e_complete_h64_s1 | 0.812 | 0.735 | 0.751 | 0.977 | 0.888 | 0.90 | 0.730 | 0.738 | 0.734 |
| n20_gatv2e_complete_h64_s2 | 0.795 | 0.725 | 0.742 | 0.976 | 0.886 | 0.80 | 0.681 | 0.773 | 0.724 |
| n20_graphconv_complete_h64_s0 | 0.777 | 0.704 | 0.727 | 0.969 | 0.842 | 0.80 | 0.654 | 0.766 | 0.706 |
| n20_graphconv_complete_h64_s1 | 0.780 | 0.706 | 0.726 | 0.970 | 0.845 | 0.80 | 0.649 | 0.774 | 0.706 |
| n20_graphconv_complete_h64_s2 | 0.791 | 0.713 | 0.732 | 0.973 | 0.855 | 0.80 | 0.671 | 0.762 | 0.714 |
| n20_graphconv_complete_h64_t500_e600_s0 | 0.773 | 0.702 | 0.725 | 0.968 | 0.856 | 0.85 | 0.670 | 0.737 | 0.702 |
| n20_graphconv_complete_h64_t500_e600_s1 | 0.769 | 0.700 | 0.723 | 0.968 | 0.859 | 0.80 | 0.649 | 0.761 | 0.701 |
| n20_graphconv_complete_h64_t500_e600_s2 | 0.764 | 0.699 | 0.723 | 0.967 | 0.848 | 0.80 | 0.630 | 0.778 | 0.696 |
| n20_graphconv_complete_h64_t500_s0 | 0.653 | 0.613 | 0.679 | 0.927 | 0.693 | 0.60 | 0.541 | 0.691 | 0.607 |
| n20_graphconv_complete_h64_t500_s1 | 0.653 | 0.613 | 0.679 | 0.928 | 0.693 | 0.55 | 0.515 | 0.726 | 0.603 |
| n20_graphconv_complete_h64_t500_s2 | 0.652 | 0.613 | 0.679 | 0.928 | 0.693 | 0.60 | 0.573 | 0.653 | 0.610 |
| n20_graphconv_knn10_h64_s0 | 0.813 | 0.730 | 0.745 | 0.976 | 0.879 | 0.70 | 0.694 | 0.775 | 0.732 |
| n20_graphconv_knn10_h64_s1 | 0.807 | 0.725 | 0.739 | 0.974 | 0.871 | 0.70 | 0.670 | 0.791 | 0.725 |
| n20_graphconv_knn10_h64_s2 | 0.811 | 0.728 | 0.744 | 0.975 | 0.883 | 0.70 | 0.694 | 0.763 | 0.727 |
| n20_graphconv_knn5_h64_s0 | 0.819 | 0.736 | 0.747 | 0.968 | 0.867 | 0.60 | 0.682 | 0.801 | 0.737 |
| n20_graphconv_knn5_h64_s1 | 0.819 | 0.737 | 0.747 | 0.968 | 0.866 | 0.55 | 0.704 | 0.779 | 0.740 |
| n20_graphconv_knn5_h64_s2 | 0.829 | 0.744 | 0.755 | 0.970 | 0.887 | 0.55 | 0.726 | 0.765 | 0.745 |
