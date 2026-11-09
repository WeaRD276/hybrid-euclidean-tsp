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

## 3. After 2-opt, the neural hybrid is the best method tested

| method | n = 10 gap % | n = 20 gap % |
| --- | ---: | ---: |
| greedy edge + 2-opt | 0.28 | 1.22 |
| farthest insertion + 2-opt | 0.35 | 1.55 |
| greedy + SoftDist + 2-opt | 0.19 | 0.89 |
| **greedy + GraphConv heatmap + 2-opt** | **0.17 ± 0.00** | **0.81 ± 0.05** |

At n = 20, each of the nine GraphConv runs beats greedy + 2-opt per instance: 286 to 323 wins
against 178 to 218 losses out of 1,000, Wilcoxon p < 1e-4 for every run.

## 4. A kNN graph improves edge metrics but not tours (H1)

| n = 20, GraphConv | AP | F1 | greedy gap % | + 2-opt gap % |
| --- | ---: | ---: | ---: | ---: |
| complete graph | 0.78 | 0.71 | 6.68 | 0.81 |
| kNN, k = 10 | 0.81 | 0.73 | 6.82 | 0.78 |
| kNN, k = 5 | 0.82 | 0.74 | 7.35 | 0.81 |

Better edge classification did not give better tours. This is the first data point for RQ3.
It also means the complete-graph sum aggregation was not the main cause of the coursework
failure (finding 1 was).

## 5. GAT without edge features collapses to edge length

At n = 20 all three GAT seeds give AP 0.655 and length ratio 0.694, the same as distance only,
so the hybrid barely moves (12.36%). At n = 10, GAT is unstable across seeds (AP 0.68 to 0.79).
This is the coursework's "biased toward short edges" behaviour, now in a different model.
Whether longer training or edge features in attention fix it is the next experiment.

## 6. The best fusion is close to "rank by p"

On validation, ranking edges purely by p (the largest λ, where distance only breaks ties) is
the best setting for the complete-graph GraphConv runs at n = 20, and within 0.4 points of the
tuned λ for every other run. The weighted sum d − λ·p adds little over ranking by p, which
matters for the hybridization study (WP5). The best raw and normalised score modes are within
0.35 points of each other on every heatmap (median 0.03), so the scale mismatch between d and p was not a
problem at these sizes.
