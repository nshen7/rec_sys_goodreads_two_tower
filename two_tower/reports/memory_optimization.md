# Memory Optimization — Two-Tower Training

## Problem

After running `build_confirmed_neg_index` and `prepare_data`, only ~2 GB RAM
remained out of 15 GB, leaving no headroom for DataLoader workers during training.

---

## Root Causes

### 1. Missing `del df` in `build_confirmed_neg_index`
The full confirmed negatives DataFrame was never freed after building the index.
It stayed alive in the kernel for the entire session.

### 2. Dataset arrays stored as int64
`history_item_ids` (29.7M rows × 10 steps) at 8 bytes/element = **~2.4 GB** for
one array alone. `user_ids` and `target_item_ids` added another ~475 MB combined.
All values fit comfortably in int32 (max item_id < 2.26M, max user_id < 876K).

### 3. All dataset arrays held fully in RAM
38M training rows + 8.5M val rows were loaded eagerly into contiguous numpy arrays,
consuming the full ~6 GB regardless of how much of the data was actually accessed.

---

## Fixes Applied

### Fix 1 — Free the confirmed negatives DataFrame
**File:** `src/data/sampler.py`

Added `del df` and `gc.collect()` immediately after building the index so the
DataFrame is released before the function returns.

```python
# before
index[int(user_id)] = group["item_id"].tolist()
return index

# after
index[int(user_id)] = group["item_id"].tolist()
del df
gc.collect()
return index
```

---

### Fix 2 — Downcast integer arrays to int32
**File:** `src/data/loaders.py`

Changed dtype from `int64` to `int32` for `user_ids`, `target_item_ids`, and
`history_item_ids` in both `__init__` and `from_parquet`. The tensors produced
in `__getitem__` are still cast to `torch.long` (int64) on-the-fly for embedding
lookups — the cast is per-row so cost is negligible.

| Array | Old dtype | New dtype | Savings |
|---|---|---|---|
| `user_ids` | int64 | int32 | ~119 MB |
| `target_item_ids` | int64 | int32 | ~119 MB |
| `history_item_ids` | int64 | int32 | ~1.19 GB |
| **Total** | | | **~1.7 GB** |

---

### Fix 3 — Memory-mapped arrays
**Files:** `src/data/loaders.py`, `configs/baseline.yaml`

Added `mmap_dir: "artifacts/mmap"` to the config. `TwoTowerDataset.from_parquet`
now supports a `mmap_dir` parameter:

- **First run:** builds arrays from GCS as before, saves them as `.npy` files
  under `artifacts/mmap/train/` and `artifacts/mmap/val/`, then immediately
  replaces the in-memory arrays with `mmap_mode='r'` versions to free RAM.
- **Subsequent runs:** skips GCS entirely, loads `.npy` files with
  `mmap_mode='r'` directly.

With mmap, the OS loads only the 4 KB pages that are actually touched by each
batch. A batch of 2048 rows accesses a tiny fraction of the 38M-row arrays, so
active RAM usage stays low instead of pinning the full ~6 GB.

**Files saved:**
```
artifacts/mmap/
  train/
    user_ids.npy
    target_item_ids.npy
    sample_weights.npy
    history_item_ids.npy
    history_item_weights.npy
  val/
    (same structure)
```

---

## RAM Budget (15 GB machine)

| Component | Before fixes | After fixes |
|---|---|---|
| Dataset arrays (train + val) | ~7.5 GB (eager) | ~0.5–1 GB (mmap, pages on demand) |
| `item_cat_feats` / `item_num_feats` | ~154 MB | ~154 MB (shared memory, unchanged) |
| DataLoader worker overhead | — | ~0.5–1 GB (4 workers) |
| Prefetched batches | — | ~200 MB |
| Python / kernel overhead | ~0.5 GB | ~0.5 GB |
| **Available for training** | **~2 GB** | **~10+ GB** |
