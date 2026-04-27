## What conclusions should I reach while reading

Why do many neural TSP papers predict a full tour instead of edge labels?
What makes full-tour prediction difficult?
Why might edge classification be a simpler and more interpretable task?
How can edge scores be integrated into a classical heuristic?

Those four questions are basically the seed of my “research motivation” section.

## Important observation about class imbalance

In TSP tour over n nodes only n undirected (2*n directed) edges are positive. 
So, 2*n/(n*(n-1)) = 2/(n - 1).
For n = 12 it's about 18.2%, which becomes less with larger n. It means, predicting all zeros will look "accurate".
This means later in training I should probably use:
- weighted BCE loss
- or at least monitor precision/recall, not just accuracy

## What to freeze now

Dataset representation:
- node features: 2D coordinates
- graph: complete directed graph obtained from undirected Euclidean TSP
- edge features: [dx, dy, dist]
- labels: binary edge membership in optimal tour
- implementation: each undirected edge represented by two directed edges with identical labels

## Suggested concrete dataset object

For one instance:
{
    "points": points,              # [n, 2]
    "dist": dist_matrix,           # [n, n]
    "edge_index": edge_index,      # [2, E]
    "edge_attr": edge_attr,        # [E, 3]
    "edge_label": edge_label,      # [E]
    "path": optimal_path           # optional, useful for debugging
}

and for PyG:
Data(
    x=x, 					# node features
    edge_index=edge_index,  # graph connectivity
    edge_attr=edge_attr,    # edge features
    y=edge_label 			# edge labels
)

Node features x:
shape - [n, 2]
x[i] = [x_i, y_i]

Edge index edge_index:
Shape - [2, n*(n-1)]
Example for n = 4: 
edge_index = [
  [0,0,0,1,1,1,2,2,2,3,3,3],
  [1,2,3,0,2,3,0,1,3,0,1,2]
]

Edge features edge_attr:
Shape - [n*(n-1), 3]
edge_attr = [dx, dy, dist]

labels y:
Shape - [n*(n-1)] or [n*(n-1), 1]
where
y[e] = 1 if edge {i,j} belongs to optimal tour else 0

# Phase 3 — First Minimal Neural Model

## Goal

Build the **simplest possible neural model** that is:

- graph-based  
- predicts **edge labels**  
- trainable with **binary cross-entropy (BCE)**  
- easy to explain in coursework  

---

## Model Overview

Design a **3-part architecture**:

1. Node encoder  
2. Message passing  
3. Edge classifier  

This directly matches my dataset structure:

- Nodes → coordinates `x ∈ ℝ^{n×2}`
- Edges → features `[dx, dy, dist]`
- Labels → binary (`edge ∈ optimal tour`)

Graph construction already comes from `TSPGraphData` :contentReference[oaicite:0]{index=0}.

---

## Step 1 — Node Encoder

### Input
x: [n, 2]

Each node is just a 2D point.

### Architecture

Use a small MLP:
2 → 64 → 64

### Output
h: [n, h]

where `h = 64`

### Intuition

- Maps raw coordinates into a **learned embedding space**
- Allows the model to capture geometric structure beyond raw `(x, y)`

---

## Step 2 — Message Passing

Apply **graph convolution layers** over the full directed graph.

### Recommended choice

- `GraphConv` or `GCNConv`

Start with:
2 layers
hidden size = 64

### Structure
h → GraphConv → ReLU → GraphConv → h'

### Why not attention yet?

- `GATConv` is more complex  
- unnecessary for the first baseline  
- harder to explain in coursework  

### Intuition

- Each node embedding becomes aware of:
  - nearby nodes
  - global graph structure  
- This is critical because TSP is **global**, not local

---

## Step 3 — Edge Classifier

For each directed edge `(i, j)`:

### Input features
h_i (source node embedding)
h_j (target node embedding)
dx, dy, d (edge features)

From my dataset:

- edge features are built as `[dx, dy, dist]` :contentReference[oaicite:1]{index=1}

### Edge representation
[h_i || h_j || dx || dy || dist]

### Classifier

Small MLP:
(2h + 3) → 64 → 1

### Output

- Single **logit per edge**
- Shape:
[E]

---

## Training Objective

Use:
BCEWithLogitsLoss

Why?

- Binary classification per edge  
- Matches labels `y ∈ {0,1}`  
- Numerically stable  

---

## Why This Model is Correct for my Project

### 1. Matches my research idea

- I am predicting **edge importance**, not full tours  
- Labels come directly from optimal solution (`isOpt`) via Held-Karp :contentReference[oaicite:2]{index=2}  

---

### 2. Clean separation of concerns

- Neural network → predicts edge scores  
- Classical algorithm → reconstructs tour  

This is exactly my intended hybrid approach.

---

### 3. Simpler than full-tour prediction

Instead of:
model → permutation of nodes

I do:
model → edge probabilities

Which is:

- easier to train  
- easier to debug  
- easier to explain  

---

### 4. Fully compatible with the pipeline

- Works directly with `TSPGraphData`
- Uses existing dataset generator :contentReference[oaicite:3]{index=3}  
- Compatible with current evaluation setup :contentReference[oaicite:4]{index=4}  

---

## Summary

Final architecture:
Node MLP (2 → 64 → 64)
↓
2× GraphConv layers
↓
Edge MLP ([h_i || h_j || edge_attr] → 64 → 1)
↓
Logits per edge
↓
BCE loss

---

## Next Step

After this model:

- train on small dataset (e.g. 1000 graphs)
- compare against baselines:
  - all zeros :contentReference[oaicite:5]{index=5}  
  - shortest edges :contentReference[oaicite:6]{index=6}  

Then iterate:
- deeper GNN
- better edge features
- reconstruction heuristic

---

This is the **correct minimal starting point**.

## First Neural Model Result

The minimal EdgeGNN was trained on 1000 Euclidean TSP instances with n=10 and evaluated on 200 held-out test instances. The model predicts whether each directed edge belongs to the optimal tour.

Compared with the n-shortest-edge baseline, EdgeGNN improves F1 from 0.620 to 0.703. This suggests that the model learns more than simple edge length: it captures some graph-level structure useful for identifying optimal tour edges.
