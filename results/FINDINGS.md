# First findings (n = 10 and n = 20)

Setup: uniform points in the unit square, LKH reference tours, 10,000 train / 1,000 val /
1,000 test instances per size, EdgeGNN (GraphConv) and GAT with hidden size 64, 30 epochs,
3 seeds per configuration. λ and the score mode are chosen on validation. Full tables are in
`n_10/results.md` and `n_20/results.md`. Gaps are mean % over LKH with 95% bootstrap CIs.

## 1. The n = 20 failure in the coursework was undertraining

| n = 20, GraphConv, complete graph | test AP | greedy + p gap % |
| --- | ---: | ---: |
| distance only ("n shortest edges"), plain greedy | 0.651 | 12.66 |
| 500 train graphs, 30 epochs (about the coursework budget) | 0.653 | 12.63 ± 0.02 |
| same 500 graphs, 600 epochs | 0.769 | 6.94 ± 0.11 |
| 10,000 train graphs, 30 epochs | 0.783 | 6.68 ± 0.27 |

With the coursework's budget the model learns only edge length: its AP and its
"length ratio" (0.693) equal the distance-only heatmap, so validation tunes λ to 0 and the
hybrid becomes plain greedy. The same 500 graphs trained 20× longer recover almost all of the
gain, so the cause is the number of optimisation steps, not the data or the architecture.

## 2. With enough training, neural guidance roughly halves the greedy gap

| method | n = 10 gap % | n = 20 gap % |
| --- | ---: | ---: |
| greedy edge | 8.06 | 12.66 |
| greedy + kNN rank heatmap (no learning) | 7.06 | 10.65 |
| greedy + SoftDist heatmap (no learning) | 5.59 | 9.67 |
| **greedy + GraphConv heatmap** (3 seeds) | **1.74 ± 0.10** | **6.68 ± 0.27** |
| farthest insertion | 0.67 | 2.51 |

The learned heatmap beats both non-learned heatmaps, so the model learns more than
"prefer near neighbours". It does **not** beat farthest insertion, which is a much stronger
baseline than greedy edge. The thesis has to compare against it.

## 3. After 2-opt, the neural hybrid has the lowest mean gap, but not significantly below SoftDist

| method | n = 10 gap % | n = 20 gap % |
| --- | ---: | ---: |
| greedy edge + 2-opt | 0.28 | 1.22 |
| farthest insertion + 2-opt | 0.35 | 1.55 |
| greedy + SoftDist + 2-opt | 0.19 | 0.89 |
| **greedy + GraphConv heatmap + 2-opt** | **0.17 ± 0.00** | **0.81 ± 0.05** |

At n = 20, each of the nine GraphConv runs beats greedy + 2-opt per instance: 286 to 323 wins
against 178 to 218 losses out of 1,000, Wilcoxon p < 1e-4 for every run. Against
SoftDist + 2-opt, however, the difference is not significant at any size (finding 10).

## 4. A kNN graph improves edge metrics but not tours (H1)

| n = 20, GraphConv | AP | F1 | greedy gap % | + 2-opt gap % |
| --- | ---: | ---: | ---: | ---: |
| complete graph | 0.78 | 0.71 | 6.68 | 0.81 |
| kNN, k = 10 | 0.81 | 0.73 | 6.82 | 0.78 |
| kNN, k = 5 | 0.82 | 0.74 | 7.35 | 0.81 |

Better edge classification did not give better tours. This is the first data point for RQ3.
It also means the complete-graph sum aggregation was not the main cause of the coursework
failure (finding 1 was).

## 5. GAT without edge features collapses to edge length; edge features or a kNN graph fix it

| n = 20, 3 seeds | AP | greedy gap % | + 2-opt gap % | train time |
| --- | ---: | ---: | ---: | ---: |
| distance only | 0.651 | 12.66 | 1.22 | |
| GAT, complete graph (coursework model) | 0.655 | 12.36 ± 0.08 | 1.25 ± 0.02 | ~16 min |
| GAT, kNN k = 10 | 0.76 | 7.79 ± 0.15 | 0.96 ± 0.04 | ~9 min |
| GATv2 with edge features in attention, complete | 0.81 | 6.54 ± 0.60 | 0.89 ± 0.09 | ~29 min |
| GraphConv, complete (for reference) | 0.78 | 6.68 ± 0.27 | 0.81 ± 0.05 | ~6 min |

On the complete graph, plain GAT reproduces the coursework's "biased toward short edges"
behaviour in all three seeds: its AP and length ratio (0.694) equal distance only. Its
attention sees only node embeddings built from raw coordinates, so it cannot tell near
nodes from far ones. Giving attention the edge features (GATv2, `edge_dim`) or restricting
it to the 10 nearest neighbours both remove the collapse. GATv2 then reaches the best AP so
far at n = 20, but its tours are no better than GraphConv's and it trains five times slower.
A longer plain-GAT run (100 epochs) was started and stopped for time, so "more training"
is not ruled out as a fix.

Within GATv2 the seed with the highest AP (0.818) gives the worst greedy gap (7.38%), and
the one with the lowest AP (0.795) gives the best (6.07%), another case for finding 9.

## 6. The best fusion is close to "rank by p"

On validation, ranking edges purely by p (the largest λ, where distance only breaks ties) is
the best setting for the complete-graph GraphConv runs at n = 20, and within 0.3 points of the
tuned λ for every other run at every size. The weighted sum d − λ·p adds little over ranking by p, which
matters for the hybridization study (WP5). The best raw and normalised score modes are within
0.4 points of each other on every heatmap at every size (median 0.03), so the scale mismatch between d and p was not a
problem at these sizes.

# Scaling to n = 50 and n = 100

Same setup (10,000 / 1,000 / 1,000 instances, 30 epochs, 3 seeds). At n = 50 GraphConv is
trained on the complete graph and on a kNN graph with k = 10; at n = 100 only on kNN (k = 10),
because the complete graph has 9,900 directed edges per instance. Tables: `n_50/results.md`,
`n_100/results.md`.

## 7. The neural gain shrinks with n, and so does its lead over non-learned heatmaps

Mean test gap % over LKH. "GraphConv" is the better graph type at each size
(complete at n ≤ 20, kNN k = 10 at n ≥ 50), mean of 3 seeds.

| method | n = 10 | n = 20 | n = 50 | n = 100 |
| --- | ---: | ---: | ---: | ---: |
| greedy edge | 8.06 | 12.66 | 17.02 | 18.03 |
| greedy + SoftDist (no learning) | 5.59 | 9.67 | 14.09 | 16.01 |
| greedy + GraphConv | 1.74 | 6.68 | 12.72 | 14.83 |
| farthest insertion | 0.67 | 2.51 | 5.76 | 7.50 |
| greedy edge + 2-opt | 0.28 | 1.22 | 2.92 | 3.53 |
| greedy + SoftDist + 2-opt | 0.19 | 0.89 | 2.37 | 2.99 |
| greedy + GraphConv + 2-opt | 0.17 | 0.81 | 2.31 | 2.88 |

Share of the greedy gap removed by the heatmap:

| heatmap | n = 10 | n = 20 | n = 50 | n = 100 |
| --- | ---: | ---: | ---: | ---: |
| SoftDist | 31% | 24% | 17% | 11% |
| GraphConv | 78% | 47% | 25% | 18% |

This is the curve the thesis has to explain. The neural heatmap still helps greedy at n = 100
(about 620 wins against 380 losses per 1,000 instances after 2-opt, p < 1e-4 against
greedy + 2-opt), but its margin over SoftDist + 2-opt is only about 0.1 points from n = 20 on,
and finding 10 shows it is not significant.

## 8. At n = 50 the complete-graph model collapses again; the kNN model does not (H1)

| n = 50, GraphConv | AP | greedy gap % | + 2-opt gap % | train time |
| --- | ---: | ---: | ---: | ---: |
| distance only | 0.655 | 17.02 | 2.92 | |
| complete graph | 0.66 | 13.53 ± 2.44 | 2.45 ± 0.29 | ~33 min |
| kNN, k = 10 | 0.76 | 12.72 ± 0.18 | 2.31 ± 0.03 | ~6 min |

With the same budget, the complete-graph model is back at "distance only" on edge metrics
and is unstable across seeds, while the kNN model trains well and is five times cheaper.
So the complete graph does matter once n grows, even though it did not at n = 20
(finding 4). It is not yet clear whether this is the sum-aggregation effect or just a
larger budget being needed again (finding 1). A longer complete-graph run at n = 50 would
separate the two.

## 9. Edge metrics rank the heatmaps differently from tour quality (RQ3)

At n = 50 and n = 100 the kNN-rank heatmap has a higher AP than GraphConv (0.79 vs 0.76 and
0.80 vs 0.74), yet GraphConv gives clearly shorter greedy tours (12.72% vs 14.88% and
14.83% vs 16.12%). Together with finding 4, this is direct evidence that a better edge
classifier is not automatically a better guide for greedy.

# Hybridization mechanisms

Script: `experiments/mechanisms.py`; tables: `n_<N>/mechanisms.md`. Heatmaps: GraphConv
(complete graph at n ≤ 20, kNN k = 10 at n ≥ 50, 3 seeds), SoftDist, kNN rank, and plain
distance (each node's k nearest neighbours). Each neural run is compared per instance with
SoftDist under the same mechanism; p is the worst Wilcoxon p over the three seeds.

## 10. Once 2-opt runs to completion, the learned heatmap stops mattering

| fused greedy + full 2-opt, gap % | n = 10 | n = 20 | n = 50 | n = 100 |
| --- | ---: | ---: | ---: | ---: |
| SoftDist | 0.19 | 0.89 | 2.37 | 2.99 |
| GraphConv | 0.17 | 0.81 | 2.31 | 2.88 |
| neural vs SoftDist, worst-seed p | 0.93 | 0.62 | 0.95 | 0.63 |

Before 2-opt the neural heatmap beats SoftDist at every size (p < 1e-4). After a full 2-opt
the difference is not significant at any size. Local search erases what the network adds
over a simple distance-based heatmap. This agrees with Xia et al. (ICML 2024) for MCTS, and
extends it to greedy construction followed by 2-opt.

## 11. The network helps where the search is weak: tiny candidate lists

Neighbour-list 2-opt from the plain greedy tour, considering only moves that add one of each
node's top-k heatmap edges:

| candidate source | n = 20, k = 2 | k = 3 | k = 5 | n = 100, k = 2 | k = 3 | k = 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k nearest by distance | 4.92 | 2.31 | 1.30 | 12.63 | 9.00 | 5.12 |
| SoftDist | 5.07 | 2.43 | 1.28 | 12.69 | 9.02 | 5.13 |
| GraphConv | 4.41 | 2.11 | 1.29 | 12.47 | 8.86 | 5.05 |

With 2 or 3 candidates per node, the neural lists give shorter tours than SoftDist lists
(p ≤ 0.009 at n ≤ 50, p ≤ 0.05 at n = 100) and lower mean gaps than plain nearest-neighbour
lists. With 5 or more candidates the sources are within 0.1 points of each other and the
neural lists are no longer better (at n = 50, k = 8 they are slightly worse, p ≤ 0.03).
A likely reason: the 5 nearest neighbours already contain about 93% of the optimal tour edges
(R@5 in the edge tables). Starting 2-opt from the fused greedy tour gives a similar picture:
significant at k = 2 at every size, and at k = 3 everywhere except n = 20. Against
nearest-neighbour lists at k = 2, the gain shrinks with n: 0.80 points at n = 10, 0.51 at
n = 20 and 0.16 at n = 100.

## 12. Pruning the greedy edge list adds nothing

Letting greedy take only each node's top-k heatmap edges first (k = 2, 3, 5), with the rest
as a fallback, stays within 0.3 points of the fused greedy at every size. k = 2 is slightly
worse, and the largest improvement is 0.23 points (SoftDist, n = 50, k = 5). Fused greedy
with a large λ already takes edges in heatmap order, so pruning changes almost nothing.
