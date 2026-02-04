# Recommendation System Improvements

This document summarizes proposed improvements to the GraphSAGE-based recommendation system, based on analysis of the model architecture, training objectives, and alignment with research literature.

---

## 1. Rating Head: Use Concatenated Embeddings Instead of Dot Product

### Current Implementation

The `predict` method in `GraphSAGERecommender` computes a dot product of user and item embeddings, then passes this single scalar through the rating head:

```
user_emb (64-dim) · item_emb (64-dim) → 1 scalar → Linear(1, 16) → ReLU → Linear(16, 1) → Sigmoid
```

### Problem: Information Bottleneck

- **64 + 64 = 128 dimensions** of embedding information are collapsed into **1 scalar** before the MLP
- The rating head can only learn a mapping from one number to [0, 1]; it cannot recover per-dimension interaction structure
- This creates a significant information loss

### Proposed Solution

Use **concatenated embeddings** as input to the rating head:

```python
# Instead of: dot product -> (B, 1) -> rating_head
link_emb = torch.cat([user_emb_selected, item_emb_selected], dim=-1)  # (B, 128)
scores = self.rating_head(link_emb)  # rating_head: Linear(128, 32) -> ReLU -> Linear(32, 1) -> Sigmoid
```

### Benefits

- Preserves **2 × hidden_dim** dimensions for the MLP to learn from
- Allows the model to learn arbitrary interaction patterns across embedding dimensions (not just dot-product similarity)
- Aligns with **Neural Collaborative Filtering (NCF)** and **StellarGraph's HinSAGE** MovieLens demo, which use concatenation for link/rating prediction

### References

- StellarGraph HinSAGE link prediction: uses `edge_embedding_method="concat"` for user-movie rating prediction
- NCF: concatenates user and item embeddings before MLP layers

---

## 2. Dot Product vs. Rating Prediction: No Guarantee of Correlation

### Current Behavior

- **`compute_scores`** (for recommendations): uses raw **dot product** of embeddings
- **`predict_rating`**: uses `rating_head(dot_product)` to produce ratings in [1, 5]

### Problem

There is **no theoretical guarantee** that the dot product correlates with rating predictions:

1. **Non-monotonic rating head**: The rating head is an MLP (Linear → ReLU → Linear → Sigmoid) with no monotonicity constraint. Higher dot product does not guarantee higher predicted rating.

2. **Different training objectives**:
   - **BPR loss**: Optimizes raw dot product for ranking; rating head is not trained
   - **MSE loss**: Optimizes rating_head(dot_product); mapping can be non-monotonic
   - **Combined loss**: BPR and MSE pull in different directions

3. **Ranking vs. rating mismatch**: The order of items by `compute_scores` (dot product) may differ from the order by `predict_rating` (rating head output).

### Proposed Solution: Joint Combined Loss with Concatenated Rating Head

Use the **combined loss** with a clear separation of roles:

1. **Graded/Weighted BPR** operates on the **dot product**: learns to rank embeddings so that `dot(u, pos) > dot(u, neg)` when the positive item is preferred over the negative (by rating value). This trains the dot product for correct preference ordering.

2. **MSE** operates on the **rating head output**: learns the rating produced by `rating_head(concat(user_emb, item_emb))`. The rating head receives concatenated embeddings (per Improvement #1) for richer input.

3. **Joint minimization**: Both losses are minimized together. Shared embeddings receive gradients from both—BPR for ranking, MSE for rating prediction.

This approach:
- Keeps the **concatenated** rating head (richer input, better rating prediction potential)
- Ensures the **dot product** is explicitly trained for ranking via graded BPR
- Relies on **emergent correlation** between dot product and rating head output, since both depend on the same embeddings. Correlation is not architecturally guaranteed but may arise from joint optimization of related preference-based objectives.

**Trade-off**: Using concatenation (vs. dot product) as rating head input prioritizes rating prediction expressivity over guaranteed dot–rating correlation. If rankings and displayed ratings diverge in practice, consider an auxiliary loss to encourage correlation, or monitor alignment during evaluation.

---

## 3. BPR Loss: Upgrade to Graded / Weighted BPR for Explicit Ratings

### Current Implementation

- **Positive pairs**: Any (user, item) with a rating (1–5)
- **Negative pairs**: Items the user has **not** rated
- **Rating values are ignored** in the BPR loss

### Problem

The current BPR effectively predicts **"existence of rating"** rather than **"quality of preference"**:

- A 1-star movie and a 5-star movie are both treated as "positive"
- The model learns to rank both above unrated items
- A user's 1-star movie could be recommended alongside 5-star movies

This is appropriate for **implicit feedback** (clicks, purchases) but not for **explicit ratings** (1–5 scale).

### Proposed Solution: Graded / Weighted BPR

Extend BPR to respect the full rating scale:

- **Preference ordering**: 5-star > 4-star > 3-star > 2-star > 1-star > unrated
- **Pair sampling**: Sample pairs where the "positive" item has a strictly higher rating than the "negative" item
- **Optional weighting**: Weight the loss by the rating gap (e.g., 5 vs 1 penalized more heavily than 4 vs 3)

### Implementation Sketch

```
For each user u:
  1. Sample item i with rating r_i and item j with rating r_j (or j unrated)
  2. Ensure r_i > r_j (or j is unrated, treated as lowest)
  3. BPR loss: -log(σ(score(u,i) - score(u,j)))
  4. Optionally weight by w(r_i, r_j) = f(r_i - r_j) to emphasize large gaps
```

### Alternative: Threshold Approach

A simpler interim fix:

- **Positive**: ratings ≥ 4 only
- **Negative**: unrated items, or ratings ≤ 2
- Exclude 3-star ratings from BPR pairs (ambiguous preference)

### Reference

- *Using Graded Implicit Feedback for Bayesian Personalized Ranking* (RecSys 2014) — extends BPR to multiple preference levels

---

## 4. GraphSAGE Paper Alignment

### What the Paper Defines

- **Embedding generation**: Sample-and-aggregate algorithm producing node embeddings
- **Unsupervised loss**: Uses dot product \(z_u^\top z_v\) in contrastive objective
- **Downstream use**: "Can be replaced or augmented by a task-specific objective"

### What the Paper Does Not Define

- How to predict link scores or ratings from embeddings
- Whether to use dot product vs. concatenation vs. MLP for link/rating prediction

### Industry Practice

- **DGL / typical link prediction**: Often use dot product for scoring
- **StellarGraph HinSAGE MovieLens demo**: Uses **concatenation** of embeddings for the link regression layer

**Conclusion**: The choice of dot product vs. concatenation for rating prediction is an application-level design decision, not prescribed by the GraphSAGE paper. StellarGraph's concatenation approach is a reasonable and well-supported choice for explicit rating prediction.

---

## Summary: Recommended Changes

| Area | Current | Proposed |
|------|---------|----------|
| **Rating head input** | Dot product (1 scalar) | Concatenated embeddings (2 × hidden_dim) |
| **Rating head architecture** | `Linear(1, 16) → ReLU → Linear(16, 1) → Sigmoid` | `Linear(2*hidden_dim, 32) → ReLU → Linear(32, 1) → Sigmoid` |
| **BPR for explicit ratings** | All ratings = positive, unrated = negative | Graded BPR: sample pairs by rating preference (higher > lower) |
| **BPR pair sampling** | Random (rated, unrated) | Sample (item_i, item_j) where rating(i) > rating(j) |

---

## Files to Modify

- `poc/graphsage_model.py`: Rating head architecture and `predict` method
- `poc/train_graphsage.py`: BPR pair sampling logic, new `GradedBPRLoss` (or extended `BPRLoss`)
- `app/core/inference/recommender.py`: `compute_scores` unchanged (dot product for efficient retrieval); `predict_rating` uses updated model

---

## Summary of All Changes (Sections 1–3)

The improvements in Sections 1–3 form a single, consistent design. Implement them together:

**Rating head**: Use concatenated embeddings `torch.cat([user_emb, item_emb], dim=-1)` as input instead of the dot product. Update the architecture to `Linear(2*hidden_dim, 32) → ReLU → Dropout → Linear(32, 1) → Sigmoid`.

**Combined loss**: Minimize BPR and MSE jointly. BPR operates on the **dot product** to learn preference ranking; MSE operates on the **rating head output** (from concatenated embeddings) to learn rating prediction. Shared embeddings receive gradients from both losses.

**Graded BPR**: Sample pairs `(item_i, item_j)` where `rating(i) > rating(j)` (or j is unrated). The "positive" item is the higher-rated one; the "negative" is lower-rated or unrated. Enforce preference ordering 5-star > 4-star > … > 1-star > unrated. Optionally weight the loss by the rating gap.
