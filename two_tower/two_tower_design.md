# Two-Tower Retrieval Recommender (PyTorch) — Project Plan

## 0) Goal
Build a two-tower retrieval model for book recommendation:
- User tower uses user characteristics + mean-pooled embeddings of the last N interacted items.
- Item tower uses item characteristics + item embeddings.
- Train with listwise InfoNCE (in-batch negatives) and add interaction-strength weighting.
- Add hard negative mining after a working baseline.

Deliverables:
- Reproducible training pipeline (data → model → training → evaluation).
- Offline retrieval metrics (Recall@K, nDCG@K) with time-based splits.
- Saved embeddings + ANN-ready artifacts (optional).

---

## 1) Data Requirements and Canonical Tables

### 1.1 Interactions (Silver)
A single Parquet table produced by Spark:
- user_id (int)
- item_id (int)  (book_id or work_id)
- ts (timestamp)
- is_read (0/1)
- rating (float, nullable)
- interaction_strength (float)  (precomputed using your rule)
- Optional: is_shelved (0/1), shelf_type, language, etc.

Notes:
- Ensure one row per (user_id, item_id) per timestamp (dedupe policy decided upstream).
- Filter out users/items with too few interactions (e.g., >= 5) to stabilize training.

### 1.2 User Features (Optional)
Parquet table:
- user_id
- user-level features (book club, etc.)

If no features exist, use only user_id embedding.

### 1.3 Item Features (Recommended)
Parquet table:
- item_id
- categorical features: author_id, genres (multi-hot), language
- numeric features: pub_year, popularity_count, avg_rating
- Optional text embedding vector precomputed from title/description (v1 can skip)

### 1.4 Train/Val/Test Split (Gold)
Time-based per-user split:
- Train: older interactions
- Val: next interaction(s)
- Test: last interaction(s)

(Optional) Also create a “user history” view for each sample:
- For each (user, target item, ts), build last N items with timestamps < ts.

---

## 2) Data Processing Pipeline (Python)

### 2.1 Dataset Builder
Input: interactions + (optional) user features + item features  
Output: training samples with:
- user_id
- user_features (optional)
- history_item_ids: list[int] of length N (pad/truncate)
- history_item_weights: list[float] (optional; derived from interaction_strength and/or recency)
- target_item_id
- target_item_features
- sample_weight (derived from interaction_strength)

Implementation notes:
- Use a deterministic padding item id (e.g., 0) and mask it.
- Cache user histories efficiently (build per-user arrays once, then slice for each interaction).

### 2.2 Negative Candidate Pools
Maintain:
- Global item pool for random negatives.
- Per-user hard-negative pool (populated later from retrieval results).

---

## 3) Model Architecture (PyTorch)

### 3.1 Embedding Modules
- user_id_embedding: Embedding(num_users, d_id)
- item_id_embedding: Embedding(num_items, d_id)
- feature embeddings:
  - categorical embeddings (author, language, etc.)
  - multi-hot genres embedding (sum/mean of genre embeddings)
- numeric features: normalized and fed into MLP

### 3.2 Item Tower
Inputs:
- item_id embedding
- item categorical embeddings
- numeric features
(Optional) precomputed text embedding vector

Combine via:
- concatenation → MLP → item_embedding (dimension d)

### 3.3 User Tower
Inputs:
- user_id embedding
- user features (optional)
- history_item_ids (last N)

Steps:
- lookup embeddings for history_item_ids using item_id_embedding (or item tower output if you choose)
- mean pooling over history embeddings (mask padding)
- optionally weighted mean pooling using history_item_weights
- concatenate pooled_history + user_id embedding + user features → MLP → user_embedding (dimension d)

Decision: For v1, history uses item_id_embedding directly (simple and fast). Later you can experiment with using full item tower outputs.

### 3.4 Matching Function
- Similarity = dot product between user_embedding and item_embedding
- Apply L2-normalization optionally (toggle)

---

## 4) Training Objective and Strategy

### 4.1 Listwise InfoNCE (In-Batch Negatives)
- Batch contains B users with their positive target items.
- Build a B x B score matrix (each user against all items in the batch).
- Use cross-entropy style loss where the diagonal is the positive.

### 4.2 Incorporating Interaction Strength
Use interaction_strength to influence training. Implement one (or more) of these options:

Option A (recommended first): Weighted loss
- Compute a per-sample weight from interaction_strength (e.g., log1p or clipped).
- Multiply each sample’s loss by weight before averaging.

Option B: Weighted sampling
- Oversample higher-strength interactions when constructing training batches.

Option C: Both A + B
- Start with A; add B only if needed.

Also decide whether “shelved-only” interactions are positives:
- v1: include but with low weight (already encoded in interaction_strength)
- alternative: filter out weakest interactions if too noisy

---

## 5) Hard Negative Mining (Add After Baseline Works)

### 5.1 Phase 1 (Warm Start)
Train with:
- in-batch negatives only
- optional random negatives appended (if you implement cross-batch memory later)

### 5.2 Phase 2 (Hard Negatives from Retrieval)
On a schedule (e.g., every epoch or every K steps):
1) Freeze a checkpoint of the current model.
2) Build (or refresh) item embeddings for a large item subset (or full catalog).
3) For each training user sample:
   - compute user embedding
   - retrieve top M items from ANN or brute force (for a subset)
   - remove positives and already-interacted items
   - keep top H as hard negatives

Store hard negatives:
- per user_id (or per training instance)
- as a list of item_ids

### 5.3 Training with Hard Negatives
Two implementation choices:

Option A: Expand the listwise set
- For each user, score: [positive item] + [hard negatives] + [random negatives]
- Compute InfoNCE over this set (per-user listwise).

Option B: Keep in-batch as base + add extra negatives
- For each user, append hard negatives to the item set and compute logits against them too.

Start with Option A (clearer, easier).

---

## 6) Evaluation Plan

### 6.1 Offline Retrieval Metrics
On validation/test:
- For each user, query embedding is computed from history before the holdout time.
- Retrieve top K items from the candidate pool.
Metrics:
- Recall@K
- nDCG@K
- Coverage (percent users with >=1 recommendation)
- Head/tail breakdown (popularity buckets)

### 6.2 Baselines for Comparison
- ALS baseline (already implemented)
- Popularity baseline

Report:
- two-tower vs ALS on the same split and same evaluation code.

---

## 7) Training System and Engineering

### 7.1 Configuration
Use YAML/JSON config for:
- embedding dims, N history length, batch size
- optimizer/lr scheduler, weight transforms
- negative mining schedule and sizes (M, H)
- feature toggles (use_text, use_user_features, normalize)

### 7.2 Logging and Reproducibility
- Set global seeds
- Log metrics per step/epoch
- Save best checkpoint by validation Recall@K
- Save embeddings artifacts for serving

Recommended tools:
- MLflow or Weights & Biases (optional)

### 7.3 Performance Considerations
- Use DataLoader with pre-batched samples.
- Pin memory + multiple workers.
- Keep feature lookups vectorized.
- For item features, pre-build tensors keyed by item_id for fast indexing.

---

## 8) Milestones (Suggested Order)

### Milestone 1: Data + Evaluation Harness
- Produce train/val/test splits with user histories (last N items).
- Implement Recall@K and nDCG@K evaluation on a sampled candidate set first.

### Milestone 2: Two-Tower v1 (ID + basic features)
- Implement model + listwise InfoNCE with in-batch negatives.
- Add weighted loss using interaction_strength.
- Verify metrics beat popularity baseline.

### Milestone 3: Add Item/User Features
- Add author/genre/language embeddings and numeric features.
- Compare ablations: ID-only vs +features.

### Milestone 4: Hard Negative Mining v1
- Implement periodic embedding refresh + ANN retrieval (FAISS recommended).
- Train with per-user hard negatives and measure lift.

### Milestone 5 (Optional): Text Embedding Hybrid
- Precompute text embeddings from book metadata.
- Add to item tower and re-run training + mining.

---

## 9) Files / Modules (Suggested Structure)

- `configs/`
  - `baseline.yaml`
  - `hardneg.yaml`
- `src/data/`
  - `loaders.py` (read parquet, build tensors)
  - `history_builder.py` (last N items)
  - `sampler.py` (weighted sampling, negative pools)
- `src/models/`
  - `two_tower.py` (user/item towers)
  - `embeddings.py` (feature embeddings)
- `src/train/`
  - `train.py` (main loop)
  - `losses.py` (InfoNCE, weighted loss)
  - `hard_negative_miner.py`
- `src/eval/`
  - `retrieval_eval.py` (Recall@K, nDCG@K, coverage)
- `scripts/`
  - `run_train.sh`
  - `build_item_index.py` (FAISS)
  - `mine_hard_negatives.py`
- `README.md` (setup, commands, results table)

---

## 10) Key Design Decisions to Document (with Alternatives)
- Use book_id vs work_id as item_id (work_id reduces edition noise)
- History encoding: mean pooling vs weighted mean vs GRU/Transformer
- Strength usage: weighted loss vs weighted sampling vs both
- Negatives: in-batch only vs random + hard negatives
- Item representation: ID-only vs metadata features vs +text embeddings
- Retrieval: brute force (small) vs FAISS/ANN (large)

End state: a reproducible two-tower retrieval system with strength-aware training and hard negative mining, benchmarked against ALS.
