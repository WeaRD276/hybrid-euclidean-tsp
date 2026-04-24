1. Why do many neural TSP papers predict a full tour instead of edge labels?
=============================================================================

The main reason is that TSP is naturally a **sequential decision problem**: a solution is a permutation of the cities, and early neural approaches were built exactly to model “input graph/set of points → output permutation.” Pointer Networks were explicitly designed for problems where the output tokens are **positions in the input**, and for TSP they represent the solution as a permutation of node indices corresponding to the optimal tour.

The 2019 attention model keeps the same overall view: it defines a policy over permutations

pθ(π∣s)=∏t=1npθ(πt∣s,π1:t−1),p_\theta(\pi\mid s)=\prod_{t=1}^n p_\theta(\pi_t\mid s,\pi_{1:t-1}),pθ​(π∣s)=t=1∏n​pθ​(πt​∣s,π1:t−1​),

so the model constructs the route one node at a time. The paper emphasizes that the decoder outputs a sequence of input nodes and conditions each step on the partial tour already built.

The survey paper classifies these TSP works as **end-to-end** methods: the ML model is trained to output a complete solution directly from the instance, rather than just auxiliary information. It specifically mentions Pointer Networks, Bello et al., and Kool & Welling as examples of this full-solution approach.

So, across the three papers, the common motivation is:

*   the TSP solution is a **tour/permutation**, so predicting a tour matches the problem statement directly

*   autoregressive neural architectures make it natural to build that permutation step by step

*   this allows training the model as a **constructive heuristic**, i.e. a neural policy that directly outputs a feasible route candidate.


* * *

2. What makes full-tour prediction difficult?
==============================================

There are several layers of difficulty.

2.1 It is a combinatorial structured output problem
---------------------------------------------------

A tour is not just a collection of local yes/no decisions. It is a **global object** with hard constraints:

*   each city must be visited exactly once

*   the solution must form a single Hamiltonian cycle

*   local choices interact strongly with future choices


Pointer Networks were introduced precisely because ordinary seq2seq models with a fixed output vocabulary do not fit problems where the output dictionary depends on input size. For TSP, the valid action space at each step is “which unvisited node to choose next,” which depends on the instance and on the current partial tour.

2.2 Sequential errors accumulate
--------------------------------

When the model predicts a tour autoregressively, each decision changes the future state. A bad early choice can force a poor final route even if later decisions are locally sensible. The attention model explicitly conditions each next choice on the partial solution π1:t−1\pi_{1:t-1}π1:t−1​, which is powerful but also means mistakes propagate through the decoding process.

2.3 Feasibility is nontrivial
-----------------------------

The survey stresses that direct ML output does **not guarantee feasibility**. For TSP, making a network output a valid permutation is already a nontrivial architectural challenge, which is why papers use pointer mechanisms or special permutation-producing layers.

The 2019 attention paper gives a concrete example: without constrained decoding, the model can output **invalid tours**, such as repeating cities or skipping destinations. For larger nnn, this becomes much worse. They therefore constrained beam search to consider only valid tours in TSP experiments.

2.4 Supervised targets can be awkward
-------------------------------------

The survey points out that supervised imitation for combinatorial outputs can be inconvenient because:

*   exact optimal labels may be expensive to generate

*   multiple tours can be equally good or nearly equally good

*   imitation binds the learner to one expert decision sequence, even when many alternatives are valid.


This is especially relevant for full-tour prediction, because the model is asked to match one particular sequence, even though TSP often admits many near-equivalent local choices.

2.5 Scaling to larger instances is hard
---------------------------------------

All three papers suggest, directly or indirectly, that full-tour neural solvers degrade as problem size grows. Pointer Networks showed some generalization, but TSP extrapolation was limited compared with easier geometric tasks.

The survey explicitly notes that learned TSP methods tend to degrade on larger instances.

So full-tour prediction is difficult because it combines:

*   structured combinatorial output

*   hard feasibility constraints

*   compounding sequential error

*   expensive or ambiguous supervision

*   scaling/generalization challenges


* * *

3. Why might edge classification be a simpler and more interpretable task?
===========================================================================

This is not the main task studied in those papers, but the comparison becomes clear when you read them together.

3.1 It breaks a hard global problem into local predictions
----------------------------------------------------------

Instead of asking the model to output an entire Hamiltonian cycle, edge classification asks for something like:

f(i,j)→score that edge (i,j) is useful / optimal / promising.f(i,j) \to \text{score that edge }(i,j)\text{ is useful / optimal / promising}.f(i,j)→score that edge (i,j) is useful / optimal / promising.

That is a much simpler output structure than a full permutation. The model is not forced to satisfy all tour constraints by itself. The survey argues more broadly that ML can be especially useful when it is used to **augment** a classical CO algorithm with meaningful predictions, rather than replace the solver entirely.

In that sense, edge classification fits the “ML provides useful information” paradigm better than end-to-end tour generation.

3.2 It is easier to supervise
-----------------------------

For edge labels, supervision is straightforward if you have optimal tours: each edge is either on the optimal tour or not. That avoids some of the difficulties of sequence targets:

*   no arbitrary decoding order

*   no teacher forcing mismatch

*   no need to encode one specific permutation among many equivalent tour representations


The survey highlights that imitation learning uses expert actions as labels, but also notes the limitation that these labels may reflect arbitrary expert choices. For edges, the target is often more stable and local than an entire decoding sequence.

3.3 It is more interpretable
----------------------------

Edge scores are easy to inspect. You can visualize:

*   which edges got high probability

*   whether the model favors short geometric connections

*   whether it highlights boundary edges, crossing-avoiding edges, or locally plausible substructures


This aligns with the survey’s view that ML can help **extract intuition and knowledge about algorithms**, not just produce final answers.

By contrast, a full-tour decoder gives one final route, but it is often harder to understand whether the model “believed” several alternatives were plausible, or which local structures drove the decision.

3.4 It separates learning from feasibility
------------------------------------------

The survey argues that exact or reliable CO often benefits from combining ML with classical algorithms because ML outputs do not naturally obey combinatorial constraints. Edge classification does exactly that separation:

*   ML handles **estimation / ranking / guidance**

*   classical heuristic handles **feasibility and search**


That is methodologically cleaner and often safer.

So edge classification can be simpler and more interpretable because it is:

*   more local

*   easier to label

*   easier to visualize

*   not responsible for enforcing the whole tour structure on its own


* * *

4. How can edge scores be integrated into a classical heuristic?
=================================================================

This is exactly where your approach becomes attractive.

The survey’s strongest general recommendation is that ML should often be used **alongside optimization algorithms**, where a master algorithm repeatedly queries an ML model for decisions, while the classical framework preserves structure and validity.

Edge scores fit this extremely well. Here are the main integration patterns.

4.1 Candidate edge pruning
--------------------------

Use the model to assign a score s_{ij} to each edge. Then:

*   keep only the top-k neighbors per node

*   or drop very low-score edges


This reduces the search space for a downstream heuristic or exact solver. Conceptually, this is similar to how the survey describes ML being used to provide meaningful properties or decisions that help the optimizer work faster.

4.2 Bias a constructive heuristic
---------------------------------

In a nearest-neighbor / insertion / greedy construction algorithm, replace the handcrafted choice rule with one informed by edge scores.

For example, from current node u, choose next node v using a score such as

α⋅MLScore(u,v)−β⋅d(u,v),\alpha \cdot \text{MLScore}(u,v) - \beta \cdot d(u,v),α⋅MLScore(u,v)−β⋅d(u,v),

subject to feasibility constraints.

This is very close in spirit to the survey’s description of learned heuristics: the master algorithm maintains the partial solution, while ML decides the next move.


4.3 Rank move candidates in local search
----------------------------------------

In 2-opt, 3-opt, or edge-exchange local search:

*   use edge scores to prioritize which exchanges to evaluate

*   prefer adding high-score edges and removing low-score edges


This can dramatically reduce wasted search effort while leaving the local-search mechanism itself classical.

4.4 Guide beam search or branching
----------------------------------

If you construct multiple partial tours in parallel, edge scores can define branching priorities. For each partial state, candidate extensions can be ranked by learned edge likelihood. This preserves the classical search structure but makes it more informed.

4.5 Warm-start an exact or hybrid solver
----------------------------------------

You can turn edge scores into:

*   a promising initial tour

*   a sparse candidate graph

*   branching priorities or penalties


Then let a classical solver refine or certify. This is consistent with the survey’s argument that ML is often most valuable when it helps a traditional optimization method rather than fully replacing it.

* * *

5. Where your method fits in this picture
==========================================

From the survey’s taxonomy, your edge-classification idea is **not** a pure end-to-end constructive solver. It belongs much more naturally to:

*   **learning heuristics**

*   **ML alongside optimization algorithms**

*   possibly **supervised imitation** if trained on optimal-edge labels

*   or **reinforcement learning** if trained through downstream solution quality


That makes it different from Pointer Net and the 2019 attention model, which are primarily **constructive full-tour predictors**.

A nice thesis-level positioning is:

> Full-tour neural models try to solve TSP directly as sequence prediction, while edge classification learns a more local, interpretable signal that can guide a classical heuristic responsible for assembling a feasible tour.

* * *

6. Condensed answer
====================

Many neural TSP papers predict a full tour because TSP is naturally represented as a permutation problem, and architectures like Pointer Networks and attention decoders were designed to output permutations directly, one node at a time.

This makes the method an end-to-end constructive heuristic.

However, full-tour prediction is difficult because the output is a globally constrained combinatorial object: feasibility is hard, sequential errors accumulate, supervision can be ambiguous, and performance degrades as instance size grows.

Edge classification can be simpler and more interpretable because it converts the task into local predictions about which connections are promising, avoids forcing the network to produce an entire Hamiltonian cycle by itself, and yields scores that are easy to inspect and visualize. This also matches the survey’s recommendation that ML often works best when augmenting a classical optimization procedure rather than replacing it entirely.

Edge scores can then be integrated into classical heuristics by pruning candidate edges, biasing greedy or insertion-based construction, ranking local-search moves, guiding beam search, or warm-starting exact/hybrid solvers.
