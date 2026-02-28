# Data Split Pipeline — Summary Report

## Overview

A PySpark pipeline that processes raw Goodreads interaction data and produces train/val/test splits for an implicit ALS recommender system. Runs on Google Cloud Dataproc, reading from and writing to GCS.

---

## Pipeline Steps

**1. Load**
Three parquet tables are loaded: deduplicated interactions, user ID map, and book ID map.

**2. Join & select**
The interactions table is inner-joined with both ID maps (broadcast joins for efficiency), resolving raw string IDs to compact integer IDs. Output columns: `user_id`, `book_id`, `is_read` (0/1), `rating` (0–5), `date_added`.

**3. Timestamp parsing**
`date_added` is parsed as `"EEE MMM dd HH:mm:ss Z yyyy"`. If the primary format fails on more than 50% of rows, a `unix_timestamp` fallback is attempted. Rows with unparseable timestamps are dropped.

**4. User-level sampling (Bernoulli)**
`SAMPLE_PCT`% (default 20%) of users are randomly sampled with `seed=42`. All interactions belonging to sampled users are retained — interaction history is never truncated mid-user.

**5. Iterative cold-start filtering**
Users with fewer than `MIN_USER_INTERACTIONS` (5) interactions and books with fewer than `MIN_BOOK_INTERACTIONS` (5) interactions are removed in alternating passes until the dataset stabilises. Checkpointing every 2 iterations prevents lineage explosion.

**6. Per-user temporal split**
Each user's interactions are ranked chronologically. Split boundaries are applied per-user by rank fraction:

| Split | Interactions kept | Approximate share |
|-------|-------------------|-------------------|
| Train | first 70% by time | ~70%              |
| Val   | next 20% by time  | ~20%              |
| Test  | last 10% by time  | ~10%              |

Every user appears in all three splits. This is a **leave-last-k (proportional)** strategy, also known as per-user temporal splitting.

**7. Validation**
Asserts that no user in val or test is absent from train — guaranteed by construction but verified explicitly to catch edge cases (e.g. a user whose entire history fell into a single bucket due to ties).

**8. Output**
Three parquet datasets written to `gs://.../splits_sample_20pct/{train,val,test}` with schema `(user_id: int, book_id: int, is_read: int, rating: int)`. Row counts are verified by re-reading from GCS.

---

## Design Decisions

- **User-level sampling** (not interaction-level) preserves each user's full history, keeping activity distributions representative.
- **Per-user temporal split** (not global) ensures every user contributes evaluation signal, avoiding the scenario where active users dominate val/test while sparse users appear only in train.
- **Broadcast joins** for the small ID maps avoid shuffle overhead on the large interactions table.
- **Iterative k-core filtering** produces a dense, mutually connected user–book graph, which is important for ALS convergence and metric reliability.
