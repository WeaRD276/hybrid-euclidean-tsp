# tsp-edge-gnn

Hybrid approach for the Traveling Salesman Problem using Graph Neural Networks.

Instead of predicting the whole route directly, the model predicts which graph edges are likely to belong to the optimal tour.  
These predictions are then combined with a classical greedy heuristic to construct the final route.

The project explores a simple and interpretable alternative to end-to-end neural TSP solvers using:
- PyTorch
- PyTorch Geometric
- Graph Neural Networks
- hybrid optimization methods