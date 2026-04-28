### Threshold Sensitivity

The model was evaluated across a range of classification thresholds.

Results show a clear precision–recall trade-off:
- 0.05 <= threshold <= 0.30 - high recall but low precision, average F1 is 0.5
- 0.30 < threshold <= 0.70 - medium recall and accuracy, average F1 is 0.65
- 0.70 < threshold <= 0.95 - high precision but low recall, average F1 is 0.4
- Best F1(≈0.69) is achieved around threshold ≈ 0.6

This confirms that the model outputs well-calibrated probabilities.

### Comparison with Shortest-Edge Baseline
The shortest-edge baseline relies purely on geometric distance, selecting the smallest edges without considering global structure.

In contrast, EdgeGNN achieves higher F1 by:
- selecting edges that are not strictly shortest, but rather that contribute to consistent global structure
- sometimes avoiding locally optimal but globally incorrect edges

This demonstrates that the model learns **structural properties beyond edge length**.

### Distance Distribution Analysis

I compared average edge distances:

- All edges: 0.527008
- Optimal edges: 0.292235
- Baseline edges: 0.206243
- EdgeGNN edges: 0.248249

The baseline strongly prefers very short edges, resulting in an average distance significantly lower than optimal.

EdgeGNN predictions have higher average distance than the baseline and are closer to optimal edges. This indicates that the model relaxes the “shortest-edge bias” when necessary, allowing longer edges that are required for forming valid global structures.

### Visual Inspection of Predictions
Figures show that:
- the baseline produces disconnected or incorrect structures (obviously)
- EdgeGNN predictions align significantly better with the optimal tour
- Many correct edges are identified

![predictions](plots/graphs/graph_000_gt.png)
![predictions](plots/graphs/graph_000_pred.png)
![predictions](plots/graphs/graph_000_overlay.png)

![predictions](plots/graphs/graph_001_gt.png)
![predictions](plots/graphs/graph_001_pred.png)
![predictions](plots/graphs/graph_001_overlay.png)

![predictions](plots/graphs/graph_002_gt.png)
![predictions](plots/graphs/graph_002_pred.png)
![predictions](plots/graphs/graph_002_overlay.png)

### Key Insights
- The model is able to identify and correct many edges that are required for global structure
- The model is able to identify and correct edges that are not strictly shortest
- The model is able to identify and correct edges that are not strictly optimal

The model does not simply learn edge length.
Instead, it learns contextual **edge importance conditioned on the entire graph**.
This is achieved through message passing, allowing node embeddings to encode global structure.


### Limitations
The model still has important limitations:
- It does not enforce tour constraints (degree = 2, connectivity)
- It struggles with long-range dependencies
- Predictions alone do not form valid tours

This suggests that the current model should be combined with a classical heuristic for final solution construction.
