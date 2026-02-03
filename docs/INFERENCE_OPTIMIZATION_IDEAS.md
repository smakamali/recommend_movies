# Inference Optimization Ideas

Ideas for minimizing inference overhead in the GraphSAGE recommender system. Organized by where they apply in the pipeline.

---

## 1. Graph Construction & Updates

### Current Behavior

- First request builds the full graph from the database (all users, movies, ratings).
- `add_user` triggers a **full graph rebuild** (`_rebuild_graph`).
- `add_rating` only rebuilds `edge_index` (cheap).

### Ideas

#### Incremental graph updates for new users

Instead of rebuilding the whole graph when `add_user` is called:

- Append one row to the user feature matrix.
- Extend `user_id_to_idx` / `idx_to_user_id`.
- Increment `num_users`.
- Re-build the single `Data` object by concatenating the new `x` and reusing existing edges (no full DB re-query or full feature recompute).

**Benefit:** Removes the heaviest cost when a new user is added.

#### Persist graph to disk

After building (or after a periodic refresh), serialize the PyG `Data` (and optionally id↔idx mappings) to disk. On startup, **load** this instead of rebuilding from the database every time.

**Benefit:** Faster startup and first request; refresh from DB on a schedule or when an explicit “refresh” is requested.

#### Lazy / background graph refresh

Keep using the in-memory graph for inference. When the DB has changed (e.g. new ratings):

- Refresh the graph in a background thread/worker and swap when ready, or
- Batch updates and do one graph rebuild per batch instead of per event.

**Benefit:** Inference stays fast while graph stays reasonably up to date.

---

## 2. Embedding Computation & Caching

### Current Behavior

- One **full** forward pass over the entire graph produces all user and item embeddings.
- **Any** new rating (or new user) invalidates the **entire** embedding cache, so the next recommendation request pays for a full forward pass.

### Ideas

#### Lazy cache invalidation

On `add_rating` / `add_user`, only **mark** the cache as stale; do **not** recompute immediately. Recompute on the **first** request that needs embeddings (e.g. `get_recommendations` or `refresh_recommendations`).

**Benefit:** Multiple ratings in a short time cause at most one full forward pass instead of one per rating.

#### Approximate “single-user” invalidation (optional)

When only one user adds a rating, treat the rest of the graph as fixed and **recompute only that user’s embedding** (e.g. one forward pass where only that user’s output is used, or a small subgraph / 1-hop neighborhood). Keep using cached embeddings for all other users and all items.

**Benefit:** Large reduction in work; trade-off is approximation (strictly, one new edge can affect many nodes).

#### Cache item embeddings longer

Item embeddings change only when the graph (or model) changes. So:

- Invalidate **user** embeddings when that user’s ratings change (or when any rating is added, if you don’t do single-user recompute).
- Keep **item** embeddings cached across those events and only recompute when you do a full graph rebuild.

**Benefit:** Cuts the effective cost of “recompute after new rating” in half (only user embeddings recomputed). Slight approximation.

#### Pre-warm after load

Right after loading the model, optionally call `initialize_graph` and run **one** `generate_embeddings` (and store in cache).

**Benefit:** First real recommendation request hits the cache and avoids the first full forward pass.

---

## 3. Recommendation Scoring (Top-N)

### Current Behavior

- For one user: `scores = user_emb[user_idx] @ item_emb.T` → O(num_items × hidden_dim).
- Then filter (already rated, low-rated) and sort to get top-N.

### Ideas

#### Approximate nearest neighbors (ANN) when catalog is large

For large `num_items` (e.g. hundreds of thousands or more), replace “score vs all items” with ANN (e.g. FAISS, HNSW) over **item_emb**. Query with `user_emb[user_idx]` to get top-K candidates, then apply your filters and return top-N.

**Benefit:** For 1.6k movies it may be overkill; for 100k+ items it reduces compute and memory bandwidth.

#### Pre-filter before scoring

If you have a small set of “candidate” items (e.g. by genre, recency, or popularity), score only those items instead of all items.

**Benefit:** Fewer dot products and smaller sort.

#### Cache “exclude” sets per user

“Exclude already rated” and “exclude low-rated” require the user’s ratings. Cache this set (or bitmask) per user in memory for the session (or with a short TTL).

**Benefit:** Avoids repeated DB hits during scoring/filtering.

---

## 4. Model & Runtime

### Current Behavior

- Full-precision PyTorch forward on CPU or GPU.

### Ideas

#### FP16 / mixed precision

Run the GraphSAGE forward in half precision (and keep scalers where needed).

**Benefit:** Faster on GPU and sometimes on CPU, with minimal accuracy impact.

#### Quantization (e.g. INT8)

Quantize the model (e.g. PyTorch quantization) for inference.

**Benefit:** Smaller and faster, especially on CPU; may need a small calibration set.

#### Export to ONNX / TensorRT

Export the model to ONNX (and optionally TensorRT on GPU) for optimized inference.

**Benefit:** Often faster and more portable; may require a single “example” graph or fixed-size inputs if using static shapes.

#### Smaller inference model

Train a smaller model (fewer layers, smaller hidden_dim) or distill the current model into a smaller one; use the small model at inference.

**Benefit:** Less compute per forward pass.

---

## 5. API & Request Handling

### Current Behavior

- One recommendation request → one user → one use of cached (or recomputed) embeddings.

### Ideas

#### Batch recommendation endpoint

Accept multiple `user_id`s in one request and return recommendations for all. Still **one** graph forward pass; then for each user, one dot product with `item_emb` and filter/sort.

**Benefit:** Amortizes graph and embedding cost across many users (e.g. nightly jobs or batch analytics).

#### Eager graph init

After `load_model()`, immediately call `initialize_graph(session)` and optionally one `generate_embeddings()` (pre-warm).

**Benefit:** First user request doesn’t pay for graph build or first forward.

#### Health check vs “ready”

Differentiate “process up” from “model + graph loaded and warmed.” Load balancers or clients can wait for “ready” before sending traffic.

**Benefit:** First request is always fast in production.

---

## 6. Database & Feature Work

### Current Behavior

- Graph is built by querying all users, movies, ratings and then building features and edges in Python.

### Ideas

#### Incremental user features

When adding one user, fetch only that user from DB and compute only that user’s feature vector; append to the existing user feature matrix (see “incremental graph updates” above).

**Benefit:** Avoids recomputing features for all users.

#### Materialized “rating existence” for exclusions

If “exclude already rated” is hot, keep a compact structure (e.g. set of (user_id, movie_id) or per-user bitset) updated when ratings are added.

**Benefit:** Filtering doesn’t hit the DB on every recommendation call.

---

## 7. Prioritized Summary

| Priority | Idea | Main gain |
|----------|------|-----------|
| **High** | Incremental graph update on `add_user` | Removes full rebuild on every new user |
| **High** | Lazy cache invalidation (mark stale, recompute on first request) | Fewer full forward passes after multiple ratings |
| **High** | Pre-warm graph + embeddings after model load | Fast first request |
| **Medium** | Persist/load graph from disk on startup | Fast startup and first request |
| **Medium** | Cache item embeddings across “user-only” invalidations | ~half the recompute cost when only user data changes |
| **Medium** | Batch recommendation API | Amortize cost over many users |
| **Lower** | ANN for top-N when catalog is large | Scale to very large item sets |
| **Lower** | FP16 / quantization / ONNX | Faster forward pass and smaller footprint |

---

## References

- **Inference flow:** `app/core/inference/engine.py`, `graph_manager.py`, `recommender.py`
- **Architecture:** `ARCHITECTURE_MVP.md`
- **README:** `README.md`
