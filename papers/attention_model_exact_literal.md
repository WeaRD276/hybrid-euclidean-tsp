Core idea
=========

The paper builds a **routing solution autoregressively**: given a routing instance sss (for example a TSP graph), the model defines a policy over permutations

pθ(π∣s)=∏t=1npθ(πt∣s,π1:t−1),p_\theta(\pi \mid s)=\prod_{t=1}^n p_\theta(\pi_t \mid s,\pi_{1:t-1}),pθ​(π∣s)=t=1∏n​pθ​(πt​∣s,π1:t−1​),

so it chooses the next node one step at a time until a full tour/solution is constructed.

* * *

1. Graph embedding method
==========================

Input / routing instance encoding
---------------------------------

For TSP, an instance is a graph with nnn nodes, where each node iii has features xix_i xi​. In TSP, xix_i xi​ is just the 2D coordinate of the city, and the graph is fully connected.

Initial node embedding
----------------------

Each node feature is projected linearly into a hidden embedding:

hi(0)=Wxxi+bx.h_i^{(0)} = W^x x_i + b^x.hi(0)​=Wxxi​+bx.

The paper uses hidden size dh=128d_h=128dh​=128.

Encoder architecture
--------------------

The encoder is Transformer-like, but without positional encodings, specifically so that embeddings are **invariant to input order**. It applies NNN stacked attention layers, each consisting of:

*   multi-head attention between nodes

*   node-wise feed-forward layer

*   skip connections

*   batch normalization.


The update equations are:

h^i=BN(hi(ℓ−1)+MHA(h1(ℓ−1),...,hn(ℓ−1)))\hat h_i = \text{BN}\big(h_i^{(\ell-1)} + \text{MHA}(h_1^{(\ell-1)},...,h_n^{(\ell-1)})\big)h^i​=BN(hi(ℓ−1)​+MHA(h1(ℓ−1)​,...,hn(ℓ−1)​)) hi(ℓ)=BN(h^i+FF(h^i)).h_i^{(\ell)} = \text{BN}\big(\hat h_i + \text{FF}(\hat h_i)\big).hi(ℓ)​=BN(h^i​+FF(h^i​)).

Graph embedding
---------------

After the final encoder layer, they compute a **graph embedding** as the mean of the final node embeddings:

hˉ(N)=1n∑i=1nhi(N).\bar h^{(N)} = \frac{1}{n}\sum_{i=1}^n h_i^{(N)}.hˉ(N)=n1​i=1∑n​hi(N)​.

Both the final node embeddings and this graph embedding are passed to the decoder.

Interpretation
--------------

In Appendix A, they explicitly interpret attention as **weighted message passing on a graph**: queries, keys, and values are computed from node embeddings; compatibilities define attention weights; messages are aggregated from neighbors; multi-head attention gives multiple message channels.

So the graph embedding method is:

*   encode each node from raw features

*   propagate information globally with multi-head self-attention

*   obtain contextualized node embeddings

*   summarize the whole instance by mean pooling into a graph embedding.


* * *

2. How probabilities over nodes are computed
=============================================

Step-by-step decision process
-----------------------------

At decoding time, the model selects one node at each step ttt. The decoder conditions on:

*   the encoder output

*   the partial solution already constructed

*   a problem-specific context and mask.


For TSP, once part of the tour is fixed, the remaining problem is: start from the **last visited node**, go through all unvisited nodes, and eventually return to the **first node**. Therefore the decoder context uses:

*   graph embedding

*   embedding of first selected node

*   embedding of last selected node.


Context embedding
-----------------

At step ttt, the context node embedding is formed by concatenating:

*   hˉ(N)\bar h^{(N)}hˉ(N): graph embedding

*   hπt−1(N)h_{\pi_{t-1}}^{(N)}hπt−1​(N)​: last chosen node embedding

*   hπ1(N)h_{\pi_1}^{(N)}hπ1​(N)​: first chosen node embedding


For the first step, learned placeholder vectors are used instead of first/last node embeddings.

Decoder attention
-----------------

The decoder introduces a special **context node** and performs attention from this context to the encoded nodes. Queries come from the context node, keys and values come from node embeddings:

q(c)=WQh(c),ki=WKhi,vi=WVhi.q_{(c)} = W^Q h_{(c)}, \quad k_i = W^K h_i, \quad v_i = W^V h_i.q(c)​=WQh(c)​,ki​=WKhi​,vi​=WVhi​.

The compatibility score of the current context with node jjj is

u(c)j=q(c)⊤kjdk,u_{(c)j} = \frac{q_{(c)}^\top k_j}{\sqrt{d_k}},u(c)j​=dk​​q(c)⊤​kj​​,

except infeasible nodes are masked with −∞-\infty−∞. For TSP, already visited nodes are masked.

Final probabilities
-------------------

After a multi-head “glimpse”-like decoder attention, they apply one final **single-head attention** layer to compute logits over nodes. These logits are clipped with

C⋅tanh⁡(⋅),C=10,C \cdot \tanh(\cdot), \quad C=10,C⋅tanh(⋅),C=10,

then masked, and finally turned into probabilities with softmax:

pi=pθ(πt=i∣s,π1:t−1)=eu(c)i∑jeu(c)j.p_i = p_\theta(\pi_t=i \mid s,\pi_{1:t-1}) = \frac{e^{u_{(c)i}}}{\sum_j e^{u_{(c)j}}}.pi​=pθ​(πt​=i∣s,π1:t−1​)=∑j​eu(c)j​eu(c)i​​.

So probabilities over nodes are computed by:

*   building a context from the current partial solution

*   comparing that context to all node embeddings with attention

*   masking infeasible nodes

*   softmaxing over the remaining nodes.


* * *

3. Training approach: RL vs supervised
=======================================

This paper’s main training method is **reinforcement learning**, not supervised imitation learning.

They define the objective as the expected solution cost under the model policy:

L(θ∣s)=Epθ(π∣s)[L(π)],L(\theta \mid s) = \mathbb{E}_{p_\theta(\pi \mid s)}[L(\pi)],L(θ∣s)=Epθ​(π∣s)​[L(π)],

where L(π)L(\pi)L(π) is the route cost, e.g. TSP tour length.

They optimize it with **REINFORCE**:

∇L(θ∣s)=Epθ(π∣s)[(L(π)−b(s))∇log⁡pθ(π∣s)].\nabla L(\theta \mid s) = \mathbb{E}_{p_\theta(\pi \mid s)} \big[(L(\pi)-b(s)) \nabla \log p_\theta(\pi \mid s)\big].∇L(θ∣s)=Epθ​(π∣s)​[(L(π)−b(s))∇logpθ​(π∣s)].

Baseline choice
---------------

Instead of supervised labels or a learned critic as the main method, they propose a **greedy rollout baseline**:

*   take the best model so far

*   decode greedily on the same instance

*   use that solution cost as baseline b(s)b(s)b(s).


They argue this is more effective and efficient than a critic/value-function baseline. They also explicitly state in the abstract that they train using REINFORCE with a simple greedy rollout baseline, and found it more efficient than using a value function.


So for your extraction:

*   **Not supervised** in the main method

*   **Not actor-critic as final choice**

*   **Main signal** = route cost from sampled full solution

*   **Variance reduction** = greedy rollout baseline from current best model.


* * *

4. How routing instances are encoded
=====================================

This is the concise version for your notes:

TSP
---

*   Node feature = 2D coordinate

*   Fully connected graph

*   Linear projection to node embeddings

*   Several self-attention layers contextualize each node

*   Mean-pooled graph embedding summarizes the full instance.



More generally
--------------

For other routing problems, they keep the same architecture and only adapt:

*   input features

*   mask

*   decoder context

*   objective.



For example:

*   VRP adds demand to node features and remaining capacity to context.


*   OP adds prize and remaining travel budget.


*   PCTSP adds prize, penalty, and remaining prize requirement.



That is one of the paper’s main selling points: **same neural skeleton, problem-specific mask/context**.

* * *

5. How decisions are produced step by step
===========================================

This is the core decoding logic.

Autoregressive construction
---------------------------

The model outputs a solution sequentially:

π1,π2,...,πn\pi_1, \pi_2, ..., \pi_nπ1​,π2​,...,πn​

with each step conditioned on the previous ones.

At each step
------------

1.  Start from encoder node embeddings and graph embedding.

2.  Build decoder context from global graph info + partial solution state.

3.  Compute attention from context to candidate nodes.

4.  Mask infeasible nodes.

5.  Softmax over feasible nodes.

6.  Sample or greedily choose the next node.

7.  Update the partial solution and repeat.


Why this matters
----------------

The model always conditions on the **current partial tour**, unlike non-autoregressive methods that try to predict all edges at once. This means it can adapt future decisions to what was already chosen.

* * *

6. What training signal they use
=================================

The training signal is **solution quality**, not target next-node labels.

More precisely:

*   sample a complete solution π\piπ from the policy

*   compute its final cost L(π)L(\pi)L(π)

*   compare against a baseline cost b(s)b(s)b(s)

*   reinforce actions if sampled solution is better than baseline

*   penalize them if worse.


The paper explains this explicitly:

*   if L(π)−b(s)<0L(\pi)-b(s) < 0L(π)−b(s)<0, the sampled solution is better than the greedy rollout baseline, so actions get reinforced

*   if it is worse, they get discouraged.


So the signal is:

*   **global reward/cost of the whole constructed route**

*   not per-step supervised action labels

*   not handcrafted local insertion reward.


* * *

7. Why their method is “construct full solution directly”
==========================================================

This is an important contrast in the paper.

Their model directly defines a policy over complete permutations:

pθ(π∣s)=∏tpθ(πt∣s,π1:t−1),p_\theta(\pi \mid s)=\prod_t p_\theta(\pi_t \mid s,\pi_{1:t-1}),pθ​(π∣s)=t∏​pθ​(πt​∣s,π1:t−1​),

and the decoder chooses nodes one by one until the entire solution is built.

This is “construct full solution directly” because:

*   it outputs the route itself, step by step

*   it does not output a relaxed solution, adjacency matrix, or edge scores that later need heavy reconstruction

*   it does not rely on a helper insertion heuristic to place a selected node into the tour


The paper contrasts itself with prior approaches:

*   Dai et al. train a model that outputs the order in which nodes are inserted into a **partial tour**, then a helper function inserts them at the best place.


*   Nowak et al. output an adjacency matrix and then recover a feasible tour with beam search.


*   Kaempfer & Wolf output a fractional relaxation and then use beam search to get an integer solution.



By contrast, this model is structurally similar to classical constructive heuristics like nearest neighbor: it **extends a partial feasible solution one decision at a time**. The paper even highlights nearest neighbor as a baseline that “also constructs a tour directly in order,” making it structurally similar to their model.

So “construct full solution directly” means:

*   the neural policy itself is the constructive heuristic

*   no separate combinatorial repair/placement module is required to define the actual route.


* * *

8. Why this differs from Pointer Networks
==========================================

This may help your comparison with the previous paper.

Pointer Net:

*   LSTM encoder-decoder

*   attention as pointer

*   often trained supervised first in that original paper


This paper:

*   replaces recurrence with **attention-based graph encoder/decoder**

*   makes input representation permutation-invariant

*   uses RL with greedy rollout baseline as core training method

*   still outputs a permutation, but does so through a graph-aware attention model.


They explicitly say the improvement comes from both:

*   the **Attention Model**

*   the **rollout baseline**.


* * *

Compact summary
===============

What is the graph embedding method?
-----------------------------------

A Transformer-style graph encoder:

*   linearly embed each node

*   apply several layers of multi-head self-attention + feed-forward updates

*   use the final node embeddings plus their mean as graph embedding.


How are probabilities over nodes computed?
------------------------------------------

At each decoding step:

*   build a context from graph embedding + partial-solution state

*   attend from this context to all nodes

*   mask infeasible/already used nodes

*   softmax the resulting logits to get next-node probabilities.


RL or supervised?
-----------------

Main method is **RL**, specifically REINFORCE with a greedy rollout baseline, not supervised imitation.

How are routing instances encoded?
----------------------------------

As node-feature graphs:

*   TSP uses coordinates

*   other problems add demand/prize/penalty features

*   same architecture, different context/mask/objective.


How are decisions produced?
---------------------------

Autoregressively, one node at a time, conditioned on the current partial solution.

What training signal is used?
-----------------------------

The final route cost of sampled solutions, compared against a greedy baseline rollout.

Why is it “construct full solution directly”?
---------------------------------------------

Because the model itself outputs the full route sequentially, rather than outputting edges/scores/insertions that need another procedure to turn into a route.

* * *

One-paragraph summary
=====================

The paper proposes an attention-based autoregressive policy for routing problems that encodes a routing instance as contextualized node embeddings using Transformer-style self-attention, plus a pooled graph embedding, and then decodes a solution one node at a time by attending from a context representing the current partial solution to all candidate nodes. Feasible next-node probabilities are obtained by masked attention and softmax. Unlike supervised Pointer Networks, the main training method is reinforcement learning: the model samples complete solutions, receives the final route cost as reward signal, and is optimized with REINFORCE using a greedy rollout baseline from the best model so far. The method is called a direct constructive approach because it produces the actual full solution sequentially, rather than predicting a relaxed structure, an adjacency matrix, or insertion priorities that must later be converted into a route.
