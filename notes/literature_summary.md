# Literature Review Notes

**Project:** Learning Edge Importance for Hybrid Heuristics in the Euclidean Traveling Salesman Problem

## 1. Project Overview

This project investigates the use of neural networks to assist classical heuristics for solving the Traveling Salesman Problem (TSP). Instead of using neural networks to directly construct tours, the model predicts the importance of edges in the optimal solution. These predictions are then incorporated into a hybrid heuristic that combines classical greedy strategies with learned edge scores.

The main goal is to study whether supervised edge prediction can improve the performance of simple constructive heuristics on Euclidean TSP instances.

This approach keeps the algorithmic structure of classical methods while introducing machine learning as a guiding signal.

---

## 2. Problem Definition

We consider the **Euclidean Traveling Salesman Problem (TSP)** in two dimensions.

Given a set of points:

V = {v₁, v₂, ..., vₙ}, where each point lies in the unit square [0,1]².

The distance between two points is defined using the Euclidean metric:

w(i, j) = ||vᵢ − vⱼ||₂

The objective is to find a permutation π of the vertices that minimizes the tour length:

L(π) = Σ w(πᵢ, πᵢ₊₁)

with πₙ connected back to π₁ to form a cycle.

The TSP is a classical NP-hard combinatorial optimization problem.

---

## 3. Learning Task

For small instances of the problem (n ≤ 12–15), the optimal tour can be computed using the **Held–Karp dynamic programming algorithm**.

From the optimal tour, we define labels for every edge:

y(i, j) = 1 if edge (i, j) belongs to the optimal tour
y(i, j) = 0 otherwise

The learning task is formulated as **supervised edge classification**.

A neural network model receives a graph representation of the TSP instance and predicts a probability:

p(i, j) ∈ [0,1]

representing how likely an edge belongs to the optimal tour.

Training will use **binary cross-entropy loss** over all edges in the graph.

---

## 4. Hybrid Heuristic

Instead of replacing classical algorithms, the predicted edge probabilities are used to guide a greedy construction heuristic.

For each candidate edge (i, j), we define a combined score:

score(i, j) = α · distance(i, j) − β · p(i, j)

where:

* distance(i, j) is the Euclidean edge weight
* p(i, j) is the predicted edge importance
* α and β are tunable parameters

Edges with lower scores are preferred during tour construction, while respecting standard TSP constraints (maximum degree 2 and no premature cycles).

This results in a **hybrid algorithm combining classical heuristics with neural predictions**.

---

## 5. Classical Algorithms (To Review)

The following classical approaches will be studied as baselines:

### Held–Karp Dynamic Programming

* Exact algorithm for TSP
* Time complexity: O(n² · 2ⁿ)
* Used to generate optimal labels for training

### Nearest Neighbor Heuristic

* Greedy tour construction
* Selects closest unvisited node
* Simple but often suboptimal

### 2-opt Local Search

* Improves existing tours by edge swaps
* Removes edge crossings
* Widely used in practical TSP solvers

### Christofides Algorithm (overview)

* Polynomial-time approximation algorithm
* Guarantees 1.5-approximation for metric TSP

---

## 6. Neural Approaches to TSP (To Review)

Several neural approaches to combinatorial optimization have been proposed in recent years.

Key directions include:

* Sequence-based models such as **Pointer Networks**
* Attention-based architectures for routing problems
* Graph Neural Networks for combinatorial optimization
* Reinforcement learning methods for constructing tours

The literature review will focus on how these models:

* represent graph inputs
* produce solutions or edge predictions
* are trained (supervised learning vs reinforcement learning)

---

## 7. Research Motivation

Most neural approaches to TSP attempt to directly construct a tour using complex models such as attention-based decoders or reinforcement learning.

However, these approaches often require large datasets and extensive training.

This project explores a simpler and more interpretable alternative:

**learning edge importance using supervised learning and integrating predictions into classical heuristics.**

Such a hybrid approach may combine the strengths of both paradigms:

* interpretability and structure from classical algorithms
* pattern recognition from neural networks

---

## 8. Planned Literature Sources

The following works will be reviewed:

1. Vinyals et al. — *Pointer Networks* (2015)
2. Kool et al. — *Attention Model for Solving Routing Problems* (2019)
3. Bengio et al. — *Machine Learning for Combinatorial Optimization: A Survey*
4. Research on Graph Neural Networks for combinatorial optimization

Additional papers may be included if relevant to edge prediction or neural-guided heuristics.

---

## 9. Expected Contribution

The expected contribution of this coursework is an experimental analysis of a hybrid method that combines:

* classical TSP heuristics
* supervised neural edge prediction

The study will evaluate whether predicted edge importance can improve the quality of greedy tours on Euclidean TSP instances.

## Paper: Pointer Networks

### Problem
Approaches like sequence-to-sequence or Neural Turing Machines, because of their architecture,
are limited, they require the size of the output dictionary to be fixed a priori. Because
of this constraint we cannot directly apply this framework to combinatorial problems where the size
of the output dictionary depends on the length of the input sequence.

### Input representation
Planar points sets P = {P_1, . . . , P_n} with n elements each, where
P_j = (x_j, y_j) are the cartesian coordinates from a uniform distribution in [0, 1] x [0, 1].

### Output representation
C^P = {C_1, . . . , C_n} - permutation of integers from 1 to n
representing the optimal path (or tour). Without loss of generality, always start in the first point.

### Learning type
Supervised

### Key idea
Repurposing the attention mechanism of Neural machines to create pointers to input elements.
An encoding RNN converts the input sequence
to a code that is fed to the generating network. At each step, the generating network
produces a vector that modulates a content-based attention mechanism over inputs. The
output of the attention mechanism is a softmax distribution with dictionary size equal to the length
of the input.
### Strengths
Works on variable sized inputs (yielding
variable sized output dictionaries), something the baseline models (sequence-to-sequence with or
without attention) cannot do directly. Even more impressively, they outperform the baselines on
fixed input size problems - to which both the models can be applied.

### Weaknesses
Can cause overfitting (especially in cases with low n) or build invalid tours.

### Relation to our project
Pointer Networks represent an end-to-end approach that directly predicts a full tour as a permutation. In contrast, our method does not attempt to generate a complete solution, but instead focuses on predicting edge importance, which is later integrated into a classical heuristic. This avoids the need to enforce global tour constraints within the neural model.

## Paper: Attention Model for Solving Routing Problems

### Problem
Traditional methods for solving routing problems (such as heuristics or branch-and-bound algorithms) are difficult to scale to large instances.
To learn heuristics for combinatorial optimization
problems better models and better ways
of training needed. Classically, approaches to the problems of combinatorial optimization can
be divided into exact methods, that guarantee finding optimal solutions, and heuristics, that trade
off optimality for computational cost, although exact methods can use heuristics internally and vice
versa. The authors believe that these policies can be parameterized using DNNs and work "parallel",
witout need to develop heuristic for each problem separately.

### Input representation
For all TSP instances, the n node locations are sampled uniformly at random in the unit square.
This distribution is chosen to be neither easy nor artificially hard and to be able to compare to other
learned heuristics.

### Output representation
Permutation of integers from 1 to n
representing the optimal tour.

### Learning type
Reinforcement with greedy rollout baseline

### Key idea
The model is based on the Transformer architecture, which uses an attention mechanism to represent states and make decisions in the route construction sequence.
Given an instance s defines a probability distribution pθ(π|s), from
which we can sample to obtain a solution (tour) π|s. The goal of a baseline is to estimate the difficulty of the instance s, such that it can
relate to the cost L(π) to estimate the advantage of the solution π selected by the model. With the greedy rollout as baseline b(s), the function L(π) − b(s) is negative if the
sampled solution π is better than the greedy rollout, causing actions to be reinforced, and vice
versa. This way the model is trained to improve over its (greedy) self.
Using attention instead of recurrence (LSTMs) we introduce invariance to the input order of the nodes, increasing learning efficiency. Also this enables parallelization, for increased computational efficiency 
This approach eliminates the need for recurrent networks used in previous models, such as the Pointer Network.

### Strengths
Significantly
improved results on learned heuristics for TSP and additionally learned strong (single construction) heuristics for multiple routing problems, which are traditionally solved by problem-specific
approaches. Can be a powerful starting point for learning heuristics for other
combinatorial optimization problems defined on graphs, if their solutions can be described as sequential decisions. 

### Weaknesses
Complicated to understand, require hard parameters tuning.

### Relation to our project
The Attention Model constructs full solutions autoregressively using reinforcement learning. Unlike this approach, our method uses supervised learning to predict edge-level information and delegates the construction of a valid tour to a classical algorithm, resulting in a simpler and more interpretable hybrid method.

## Paper: A survey on ML for combinatorial optimization

### Problem
Combinatorial optimization encompasses problems on graphs, routes, or resource allocation where the best solution needs to be found among a large number of discrete alternatives. Traditionally, these have been solved using heuristics or constraint-based exploration methods. Bengio et al.’s review brings together how machine learning—especially deep and reinforcement learning—can automatically learn these heuristics.

### Key idea
Learning to Construct Solutions — direct creation of solutions (e.g., neural generators for the traveling salesman problem).

Learning to Improve — models that guide or improve traditional heuristics.

Learning to Explore/Select — choosing next steps in combinatorial searches.

### Learning type
Both supervised and reinforcement learning

### Role in algorithms
- Constructive (end-to-end)
- Improvement methods
- Learning heuristics

### Relation to our project
Can be useful to enderstand the general concepts.

### Our position
Many neural TSP papers predict a full tour because TSP is naturally represented as a permutation problem, and architectures like Pointer Networks and attention decoders were designed to output permutations directly, one node at a time.

This makes the method an end-to-end constructive heuristic.

However, full-tour prediction is difficult because the output is a globally constrained combinatorial object: feasibility is hard, sequential errors accumulate, supervision can be ambiguous, and performance degrades as instance size grows.

Edge classification can be simpler and more interpretable because it converts the task into local predictions about which connections are promising, avoids forcing the network to produce an entire Hamiltonian cycle by itself, and yields scores that are easy to inspect and visualize. This also matches the survey’s recommendation that ML often works best when augmenting a classical optimization procedure rather than replacing it entirely.

Edge scores can then be integrated into classical heuristics by pruning candidate edges, biasing greedy or insertion-based construction, ranking local-search moves, guiding beam search, or warm-starting exact/hybrid solvers.

Additionally, supervised edge classification provides a stable training signal, unlike reinforcement learning approaches that rely on high-variance reward signals. This makes the learning process more predictable and easier to debug.

