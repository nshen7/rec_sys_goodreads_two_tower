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
Raw Parquet table from GCS (`goodreads_interactions_dedup`):
- user_id (str, hashed) → mapped to integer via `user_id_map`
- book_id (str) → mapped to integer `item_id` via `book_id_map`
- date_added (str, format `"EEE MMM dd HH:mm:ss Z yyyy"`) → parsed to timestamp
- is_read (bool → int 0/1)
- rating (int 0–5)

Deduplication: one row per `(user_id, item_id)` kept (most recent by timestamp).
Cold-start filter: users and items with `< 5` interactions removed iteratively until convergence.

### 1.2 User Features (Optional)
Parquet table:
- user_id
- user-level features (book club, etc.)

If no features exist, use only user_id embedding.

### 1.3 Item Features (Implemented)
Extracted from the `books` Parquet table and joined into the training splits:
- item_id (integer, from `book_id_map`)
- categorical: `book_primary_author_id` (str), `book_language` (str), `book_format` (str), `book_is_ebook` (0/1)
- top-3 genre tags: `book_top_shelves` (array\<str\>[3], right-padded with `""`)
- numeric: `book_avg_rating` (float), `book_ratings_count` (int), `book_num_pages` (int), `book_publication_year` (int)
- Optional text embedding vector precomputed from title/description (v1 can skip)

### 1.4 Train/Val/Test Split (Gold)
Per-user temporal split (70/20/10 by row count within each user):
- Train: first 70% of each user's interactions (chronological)
- Val: next 20%
- Test: last 10%

Post-split filter: val/test restricted to users and items seen in train.
Production counts (20% user sample): train 30.5M, val 8.7M, test 4.3M.

User history: for each sample, the last 10 prior interactions (by timestamp) are attached as `history_item_ids` + `history_item_weights` (padded to fixed length 10 with item_id=0, weight=0.0).

---

## 2) Data Processing Pipeline (PySpark)

Implemented in `jobs/01_data_prep.py` (Dataproc job) and prototyped in `notebooks/01_data_prep_sample_mode.ipynb`.

### 2.1 Input Sources (GCS)

| Table | Path | Description |
|-------|------|-------------|
| `goodreads_interactions_dedup` | `data/parquet/` | Raw interactions: `user_id` (str), `book_id` (str), `date_added` (str), `is_read` (bool), `rating` (int 0–5) |
| `user_id_map` | `data/parquet/` | Maps hashed string `user_id` → integer `user_id_csv` |
| `book_id_map` | `data/parquet/` | Maps string `book_id` → integer `book_id_csv` (~2.36M books) |
| `books` | `data/parquet/` | Book metadata: `average_rating`, `ratings_count`, `language_code`, `format`, `num_pages`, `publication_year`, `is_ebook`, `authors` (array), `popular_shelves` (array) |

Full dataset: 228M interactions, 876K users, 2.36M books.

### 2.2 Processing Steps

**Step 1 — User-level Bernoulli sampling** (for fast iteration)
Sample `SAMPLE_PCT`% of users (Bernoulli), keep all their interactions. Default: 20% for the Dataproc job (1% in the notebook prototype). Uses `SAMPLE_SEED=42`.

**Step 2 — ID mapping**
Join hashed string IDs → integer IDs (`user_id_csv`, `book_id_csv` aliased to `item_id`). Select core columns: `user_id`, `item_id`, `is_read` (int), `rating` (int), `date_added`.

**Step 3 — Book feature extraction**
From raw `books` parquet, extract and clean:
- `book_avg_rating` (float, nullable), `book_ratings_count` (int, nullable)
- `book_language`: COALESCE null/empty → `"unknown"`
- `book_format`: COALESCE null/empty → `"unknown"`
- `book_num_pages` (int, nullable), `book_publication_year` (int, nullable)
- `book_is_ebook`: 1 if `is_ebook == "true"` else 0
- `book_primary_author_id`: `authors[0].author_id` (string)
- `book_top_shelves`: top `TOP_SHELVES=3` shelf names from `popular_shelves`, right-padded with `""` to exactly 3 elements

**Step 4 — Timestamp parsing**
Parse `date_added` string → timestamp using format `"EEE MMM dd HH:mm:ss Z yyyy"`. Rows that fail parsing are dropped. Falls back to `unix_timestamp` if primary format fails >50% of rows. Timestamp range in production run: 1977-08-27 → 2017-11-04.

**Step 5 — Interaction strength computation**
```
interaction_strength = 1.0 * is_shelved + 2.0 * is_read + BETA * max(0, rating - 3)
```
`BETA=1.0` (CLI arg). Resulting values:
- `1.0` — shelved only, not read (or low rating)
- `3.0` — read, rating ≤ 3
- `4.0` — read, rating 4
- `5.0` — read, rating 5

Distribution in 20% sample (45.6M interactions): 1.0: 52%, 3.0: 17%, 4.0: 16%, 5.0: 16%.

**Step 5b — Split confirmed negatives**
Separate `is_read=1, rating∈{1,2}` rows from the positive table before deduplication. These are books the user finished and explicitly disliked — a higher-confidence negative signal than random non-interacted items.
- `confirmed_negatives`: `(user_id, item_id, rating)` — written to `confirmed_negatives/` after cold-start filtering (restricted to surviving training users/items).
- `positives`: all remaining rows — continue through the rest of the pipeline unchanged.

**Step 6 — Deduplication**
Keep the most recent interaction per `(user_id, item_id)` pair (ordered by `date_added_ts` desc). Removed ~0.46% of rows in production run.

**Step 7 — Cold-start filtering**
Iteratively remove users with `< MIN_USER_INTERACTIONS=5` and items with `< MIN_ITEM_INTERACTIONS=5` until convergence. Converged in 3 iterations in production, reducing from 45.9M → 43.7M interactions (161K users, 771K items).

**Step 8 — Per-user temporal split**
Order each user's interactions chronologically by `date_added_ts`, then split by position:
- Train: first 70% of each user's interactions
- Val: next 20%
- Test: last 10%

Post-split filtering: val/test rows are restricted to users and items that appear in train (no cold-start in eval). Production counts: train 30.5M (70%), val 8.7M (20%), test 4.3M (10%).

**Step 9 — User history building**
For each row, collect all *prior* interactions for that user (window excludes current row), take the last `HISTORY_LENGTH=10`, and pad to a fixed-length array:
- `history_item_ids`: `array<int>[10]`, left-padded with `PAD_ITEM_ID=0`
- `history_item_weights`: `array<float>[10]`, left-padded with `0.0` (values = `interaction_strength` of each history item)

**Step 10 — Attach book features & finalise schema**
Left-join book features on `item_id`. Select final columns (see §2.3). `interaction_strength` is renamed to `sample_weight`.

**Step 11 — Validation checks**
Five checks run automatically: (1) no nulls in critical columns, (2) all history arrays have length 10, (3) timestamp range per split, (4) sample weight distribution, (5) book feature coverage (~100% in production).

### 2.3 Output Schema

Written as Parquet to GCS (`data/two_tower_splits/{train,val,test}/` and `data/two_tower_splits/confirmed_negatives/`). Local sample splits at `data/sample_splits/`.

| Column | Type | Notes |
|--------|------|-------|
| `user_id` | int | Integer user ID |
| `target_item_id` | int | Integer item ID |
| `history_item_ids` | array\<int\>[10] | Last 10 item IDs; left-padded with 0 |
| `history_item_weights` | array\<float\>[10] | Interaction strengths of history items; left-padded with 0.0 |
| `sample_weight` | float | = `interaction_strength` of target interaction; range [1.0, 5.0] |
| `timestamp` | timestamp | `date_added_ts` of the target interaction |
| `is_read` | int | 0/1 |
| `rating` | int | 0–5 |
| `book_avg_rating` | float | Nullable |
| `book_ratings_count` | int | Nullable |
| `book_language` | string | Default `"unknown"` |
| `book_format` | string | Default `"unknown"` |
| `book_num_pages` | int | Nullable |
| `book_publication_year` | int | Nullable |
| `book_is_ebook` | int | 0/1 |
| `book_primary_author_id` | string | Nullable |
| `book_top_shelves` | array\<string\>[3] | Right-padded with `""` |

### 2.4 Key Constants

| Parameter | Default | CLI arg |
|-----------|---------|---------|
| `SAMPLE_PCT` | 20.0 (job) / 1.0 (notebook) | `--sample-pct` |
| `SAMPLE_SEED` | 42 | `--sample-seed` |
| `MIN_USER_INTERACTIONS` | 5 | `--min-user-interactions` |
| `MIN_ITEM_INTERACTIONS` | 5 | `--min-item-interactions` |
| `BETA` | 1.0 | `--beta` |
| `TRAIN_RATIO` | 0.7 | `--train-ratio` |
| `VAL_RATIO` | 0.2 | `--val-ratio` |
| `HISTORY_LENGTH` | 10 | `--history-length` |
| `PAD_ITEM_ID` | 0 | (hardcoded) |
| `TOP_SHELVES` | 3 | `--top-shelves` |

### 2.5 Negative Candidate Pools
Maintain:
- **Confirmed negatives** (`confirmed_negatives/`): `is_read=1, rating∈{1,2}` pairs, restricted to training users/items. Loaded at training time as a per-user dict for use as privileged negatives in the listwise loss. Higher confidence than random negatives (explicit dislike vs. unobserved).
- **In-batch negatives**: other items in the same batch — free from the InfoNCE formulation.
- **Random negatives**: sampled from the global item pool, optionally weighted by popularity (`book_ratings_count`).
- **Hard negatives** (Phase 2): per-user hard-negative pool populated from ANN retrieval results.

---

## 3) Model Architecture (PyTorch)

### 3.1 Embedding Modules

**ID embeddings:**
- `user_id_embedding`: `Embedding(num_users + 1, d_id)` — index 0 reserved/unused
- `item_id_embedding`: `Embedding(num_items + 1, d_id)` — index 0 = PAD token (masked in history pooling)

**Categorical embeddings** (one `nn.Embedding` per feature; index 0 = unknown/pad token):
- `author_id` (int): sentinel −1 in data → remapped to index 0 at load time
- `book_language` (string): vocab built at load time from `item_features`; null already handled by pipeline (→ `"unknown"`)
- `book_format` (string): vocab built at load time; null already handled by pipeline (→ `"unknown"`)
- `shelf_0`, `shelf_1`, `shelf_2`: the 3 slots of `book_top_shelves` treated as 3 independent categoricals; vocab built at load time from all unique shelf names; empty string `""` → index 0

**Numeric features** (5 scalars, normalized at load time; nulls → 0.0 before normalization):
- `book_avg_rating` (float, nullable) — min-max normalize over [0, 5]
- `book_ratings_count` (int, nullable) — log1p then min-max normalize
- `book_num_pages` (int, nullable) — log1p then min-max normalize
- `book_publication_year` (int, nullable) — min-max over valid range (e.g. 1800–2020); null → 0.0
- `book_is_ebook` (int, 0/1) — pass through as-is

### 3.2 Item Tower
Inputs (column names from training samples / `item_features` table):
- `item_id` embedding (d_id)
- `author_id` categorical embedding (d_cat)
- `book_language` categorical embedding (d_cat)
- `book_format` categorical embedding (d_cat)
- `shelf_0`, `shelf_1`, `shelf_2` categorical embeddings (d_cat each) — mean-pooled to one d_cat vector
- numeric feature vector (5 dims, normalized)

Combine via:
- concatenation → MLP → `item_vec` (dimension d)

(Optional) precomputed text embedding vector — skip in v1.

### 3.3 User Tower
Inputs:
- `user_id` embedding (d_id)
- `history_item_ids` (array\<int\>[10]) — look up via `item_id_embedding`

No explicit user feature vector in v1 (`user_stats` table is informational only).

Steps:
1. Look up `item_id_embedding` for all 10 history slots → shape [10, d_id]
2. Weighted mean pooling using `history_item_weights` as weights; pad positions (`item_id == 0`) have weight 0.0 so masking is automatic → shape [d_id]
3. Concatenate pooled_history + user_id_embedding → MLP → `user_vec` (dimension d)

Decision: history uses `item_id_embedding` directly in v1 (not full item tower output). Can be upgraded later.

### 3.4 Matching Function
- Similarity = dot product between `user_vec` and `item_vec`
- Apply L2-normalization optionally (toggle; recommended for training stability)

### 3.5 Null / Sentinel Handling

| Feature | Raw data | Policy at load time |
|---|---|---|
| `book_avg_rating` | nullable float | null → 0.0 |
| `book_ratings_count` | nullable int | null → 0 |
| `book_num_pages` | nullable int | null → 0 |
| `book_publication_year` | nullable int | null → fill with dataset median, then normalize |
| `author_id` | int, −1 if unknown | −1 → remap to embedding index 0 |
| `book_language` | string, `"unknown"` if null | `"unknown"` → its own vocab index (not 0) |
| `book_format` | string, `"unknown"` if null | `"unknown"` → its own vocab index (not 0) |
| `book_top_shelves` slot | `""` if fewer than 3 shelves | `""` → embedding index 0 (pad/unknown token) |

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

### Milestone 1: Data + Evaluation Harness ✓
- ✓ Produce train/val/test splits with user histories (last N items) — `jobs/01_data_prep.py`
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
