# tsp-edge-gnn

Hybrid approach for the Traveling Salesman Problem using Graph Neural Networks.

Instead of predicting the whole route directly, the model predicts which graph edges are likely to belong to the optimal tour.  
These predictions are then combined with a classical greedy heuristic to construct the final route.

The project explores a simple and interpretable alternative to end-to-end neural TSP solvers using:
- PyTorch
- PyTorch Geometric
- Graph Neural Networks
- hybrid optimization methods

## Thesis pipeline

The thesis experiments use three entry points. They replace the coursework scripts in
`training/` and `experiments/`, which are kept for reference.

```bash
pip install -r requirements.txt

# 1. Instances: uniform points + LKH reference tours (LKH matches exact Held-Karp at n <= 12)
python -m data.instances --n 10 20 --train 10000 --val 1000 --test 1000

# 2. Train an edge classifier (graph: complete or knn)
python train.py --n 20 --model graphconv --graph knn --k 10 --seed 0

# 3. Evaluate every trained run for a size against classical baselines
python evaluate.py --n 20

# Tests
python -m pytest tests
```

`evaluate.py` picks the fusion weight λ and the score mode on the validation split and
applies them once to the test split. Results go to `results/n_<N>/` and are committed:

- `results.md`: tour gaps with bootstrap CIs, per-instance win/tie/loss and Wilcoxon tests,
  plus edge metrics (AP, P@n, R@k, length ratio, F1).
- `tours.csv`, `edges.csv`, `lambda_sweep.csv`: the same numbers for plots.

Score modes for the hybrid greedy (`heuristics/scores.py`):

- `raw`: `d − λ·p`, the coursework formula.
- `norm`: `d / mean_nn_dist − λ·p`, so λ means the same thing at every n.

p(u,v) and p(v,u) are averaged before scoring.
