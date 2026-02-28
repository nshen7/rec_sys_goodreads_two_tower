"""
Two-Tower Data Loader

Responsibilities:
- Load train/val Parquet from GCS (via pandas + fsspec)
- Build categorical vocabs and numeric normalization stats from the training split
- Build pre-indexed item feature tensors (LongTensor for cat indices, FloatTensor for numerics)
- Return a TwoTowerDataset (torch.utils.data.Dataset) for use with DataLoader

Key design: item features are stored as two dense tensors indexed directly by item_id
(item_cat_feats [num_items+1, 6] and item_num_feats [num_items+1, 5]), not looked up
per-row. This is ~100x faster in __getitem__ and enables shared memory across workers.
"""

from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Any, Iterator

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset


# ── Constants ──────────────────────────────────────────────────────────────────

# Column order in item_cat_feats tensor (6 columns)
CAT_COLS = ["author_idx", "language_idx", "format_idx", "shelf_0_idx", "shelf_1_idx", "shelf_2_idx"]
N_CAT = len(CAT_COLS)  # 6

# Column order in item_num_feats tensor (5 columns)
NUM_COLS = [
    "book_avg_rating",
    "book_ratings_count",
    "book_num_pages",
    "book_publication_year",
    "book_is_ebook",
]
N_NUM = len(NUM_COLS)  # 5


# ── Chunked Parquet reader ──────────────────────────────────────────────────────

_PYARROW_BATCH_SIZE = 200_000  # rows per batch; tune down if RAM is still tight


def _iter_parquet_batches(
    path: str,
    columns: list[str],
    batch_size: int = _PYARROW_BATCH_SIZE,
) -> Iterator[pd.DataFrame]:
    """
    Yield DataFrames of at most `batch_size` rows from a Parquet path (local or GCS).

    Uses PyArrow's iter_batches so only one batch lives in memory at a time.
    The path may be a single file or a directory of part-files.
    """
    fs, fpath = fsspec.core.url_to_fs(path)
    dataset = pq.ParquetDataset(fpath, filesystem=fs)
    for frag in dataset.fragments:
        pf = pq.ParquetFile(frag.path, filesystem=fs)
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            yield batch.to_pandas()


# ── Artifact construction ──────────────────────────────────────────────────────

def build_artifacts(
    train_batches: Iterator[pd.DataFrame],
    cfg: DictConfig,
) -> dict[str, Any]:
    """
    Compute all vocab dicts and normalization stats from the training split,
    processing one batch at a time to avoid loading the full file into memory.

    Must be called with the training split only — vocabs and stats are derived
    from training data to prevent val/test leakage.

    Parameters
    ----------
    train_batches : Iterator[pd.DataFrame]
        Iterator of batches from the training Parquet (see _iter_parquet_batches).
        Each batch must contain the _ARTIFACT_COLS columns.
    cfg : DictConfig
        Full config (uses cfg.numeric for declared ranges).

    Returns
    -------
    dict with keys:
        author_vocab    : dict[str, int]   (index 0 = unknown)
        language_vocab  : dict[str, int]   (index 0 = true OOV, not "unknown")
        format_vocab    : dict[str, int]   (index 0 = true OOV, not "unknown")
        shelf_vocab     : dict[str, int]   (index 0 = "" pad)
        num_users       : int
        num_items       : int
        norm_stats      : dict[str, dict]  per-feature {min, max, use_log1p}
        pub_year_median : float            training-split median for null-fill
    """
    # Accumulators for string vocabs
    author_set: set[str] = set()
    language_set: set[str] = set()
    format_set: set[str] = set()
    shelf_set: set[str] = set()

    # Accumulators for cardinalities
    max_user_id: int = 0
    max_item_id: int = 0

    # Accumulators for numeric running min/max (post log1p where applicable)
    num_min: dict[str, float] = {col: float("inf") for col in NUM_COLS}
    num_max: dict[str, float] = {col: float("-inf") for col in NUM_COLS}
    use_log1p_map: dict[str, bool] = {}

    # Reservoir sample for pub_year median estimation (~1M values is more than enough)
    _RESERVOIR_SIZE = 1_000_000
    pub_year_reservoir: list[float] = []
    pub_year_count: int = 0

    for batch in train_batches:
        # String vocab accumulation
        author_set.update(
            batch["book_primary_author_id"].fillna("-1").astype(str).unique()
        )
        language_set.update(
            batch["book_language"].fillna("unknown").astype(str).unique()
        )
        format_set.update(
            batch["book_format"].fillna("unknown").astype(str).unique()
        )
        for shelves in batch["book_top_shelves"]:
            if isinstance(shelves, (list, np.ndarray)):
                shelf_set.update(shelves[:3])

        # Cardinalities
        max_user_id = max(max_user_id, int(batch["user_id"].max()))
        max_item_id = max(max_item_id, int(batch["target_item_id"].max()))

        # Numeric running min/max
        for col in NUM_COLS:
            if col not in batch.columns:
                continue
            col_cfg = cfg.numeric[col]
            use_log1p = bool(col_cfg.use_log1p)
            use_log1p_map[col] = use_log1p
            vals = batch[col].dropna().values.astype(float)
            if len(vals) == 0:
                continue
            if use_log1p:
                vals = np.log1p(vals)
            batch_min = float(vals.min())
            batch_max = float(vals.max())
            if batch_min < num_min[col]:
                num_min[col] = batch_min
            if batch_max > num_max[col]:
                num_max[col] = batch_max

        # Reservoir sampling for pub_year median
        py_vals = batch["book_publication_year"].dropna().values.astype(float).tolist()
        for v in py_vals:
            pub_year_count += 1
            if len(pub_year_reservoir) < _RESERVOIR_SIZE:
                pub_year_reservoir.append(v)
            else:
                j = random.randrange(pub_year_count)
                if j < _RESERVOIR_SIZE:
                    pub_year_reservoir[j] = v

    # Build vocabs from accumulated sets
    author_vocab = _build_string_vocab(pd.Series(list(author_set)), pad_value="-1")
    language_vocab = _build_string_vocab(pd.Series(list(language_set)), pad_value="__OOV__")
    format_vocab = _build_string_vocab(pd.Series(list(format_set)), pad_value="__OOV__")
    shelf_set.discard("")  # pad handled by _build_string_vocab
    shelf_vocab = _build_string_vocab(pd.Series([""] + list(shelf_set)), pad_value="")

    # Build norm_stats from running min/max
    norm_stats: dict[str, dict] = {}
    for col in NUM_COLS:
        col_cfg = cfg.numeric[col]
        data_min = num_min[col] if num_min[col] != float("inf") else 0.0
        data_max = num_max[col] if num_max[col] != float("-inf") else 1.0
        norm_stats[col] = {
            "min": col_cfg.declared_min if col_cfg.declared_min is not None else data_min,
            "max": col_cfg.declared_max if col_cfg.declared_max is not None else data_max,
            "use_log1p": use_log1p_map.get(col, bool(col_cfg.use_log1p)),
        }

    pub_year_median = float(np.median(pub_year_reservoir)) if pub_year_reservoir else 2000.0

    return {
        "author_vocab": author_vocab,
        "language_vocab": language_vocab,
        "format_vocab": format_vocab,
        "shelf_vocab": shelf_vocab,
        "num_users": max_user_id,
        "num_items": max_item_id,
        "norm_stats": norm_stats,
        "pub_year_median": pub_year_median,
    }


def _build_string_vocab(series: pd.Series, pad_value: str = "") -> dict[str, int]:
    """
    Build {token: index} mapping. pad_value → index 0.
    All other unique values → indices 1, 2, 3, ... (sorted for determinism).
    """
    unique = sorted(v for v in series.unique() if v != pad_value)
    vocab = {pad_value: 0}
    vocab.update({v: i + 1 for i, v in enumerate(unique)})
    return vocab



def save_artifacts(artifacts: dict[str, Any], path: str | Path) -> None:
    """Save artifacts dict to disk via torch.save. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts, path)


def load_artifacts(path: str | Path) -> dict[str, Any]:
    """Load artifacts dict from disk via torch.load."""
    return torch.load(path, weights_only=False)


# ── Item feature tensors ───────────────────────────────────────────────────────

def build_item_feature_tensors(
    item_batches: Iterator[pd.DataFrame],
    artifacts: dict[str, Any],
) -> tuple[Tensor, Tensor]:
    """
    Build dense item feature tensors indexed directly by item_id.

    Row 0 = PAD item (all zeros / unknown indices).
    Row i = features for item_id=i.

    Processes one batch at a time; only the first-seen row per item_id is used,
    so duplicate items across batches are ignored without buffering all rows.

    Parameters
    ----------
    item_batches : Iterator[pd.DataFrame]
        Iterator of batches from the training Parquet (see _iter_parquet_batches).
        Each batch must contain the _ITEM_FEAT_COLS columns.
    artifacts : dict
        Output of build_artifacts().

    Returns
    -------
    item_cat_feats : LongTensor  [num_items+1, 6]
        Categorical embedding indices:
        col 0: author_idx
        col 1: language_idx
        col 2: format_idx
        col 3: shelf_0_idx
        col 4: shelf_1_idx
        col 5: shelf_2_idx

    item_num_feats : FloatTensor [num_items+1, 5]
        Normalized numeric features:
        col 0: book_avg_rating
        col 1: book_ratings_count
        col 2: book_num_pages
        col 3: book_publication_year
        col 4: book_is_ebook
    """
    num_items = artifacts["num_items"]
    author_vocab = artifacts["author_vocab"]
    language_vocab = artifacts["language_vocab"]
    format_vocab = artifacts["format_vocab"]
    shelf_vocab = artifacts["shelf_vocab"]
    norm_stats = artifacts["norm_stats"]
    pub_year_median = artifacts["pub_year_median"]

    # Allocate output arrays (row 0 = PAD, stays zero)
    cat_arr = np.zeros((num_items + 1, N_CAT), dtype=np.int64)
    num_arr = np.zeros((num_items + 1, N_NUM), dtype=np.float32)

    # Track which item_ids have been filled so we don't overwrite with a later duplicate
    seen_items = np.zeros(num_items + 1, dtype=bool)

    for batch in item_batches:
        # Rename and select only needed columns, drop duplicate item_ids within batch
        items_df = (
            batch
            .rename(columns={"target_item_id": "item_id"})
            [["item_id",
              "book_primary_author_id", "book_language", "book_format",
              "book_top_shelves",
              "book_avg_rating", "book_ratings_count", "book_num_pages",
              "book_publication_year", "book_is_ebook"]]
            .drop_duplicates(subset=["item_id"])
            .set_index("item_id")
        )

        # Only process item_ids we haven't seen yet
        item_ids = items_df.index.values
        new_mask = ~seen_items[item_ids]
        if not new_mask.any():
            continue
        items_df = items_df.iloc[new_mask]
        item_ids = item_ids[new_mask]
        seen_items[item_ids] = True

        # ── Categorical features ──────────────────────────────────────

        authors = items_df["book_primary_author_id"].fillna("-1").astype(str)
        cat_arr[item_ids, 0] = authors.map(lambda v: author_vocab.get(v, 0)).values

        languages = items_df["book_language"].fillna("unknown").astype(str)
        cat_arr[item_ids, 1] = languages.map(lambda v: language_vocab.get(v, 0)).values

        formats = items_df["book_format"].fillna("unknown").astype(str)
        cat_arr[item_ids, 2] = formats.map(lambda v: format_vocab.get(v, 0)).values

        for slot_col, slot_idx in [(3, 0), (4, 1), (5, 2)]:
            shelf_vals = items_df["book_top_shelves"].apply(
                lambda x: x[slot_idx] if isinstance(x, (list, np.ndarray)) and len(x) > slot_idx else ""
            )
            cat_arr[item_ids, slot_col] = shelf_vals.map(lambda v: shelf_vocab.get(v, 0)).values

        # ── Numeric features ──────────────────────────────────────────

        num_arr[item_ids, 0] = _normalize(
            items_df["book_avg_rating"].values, norm_stats["book_avg_rating"]
        )
        num_arr[item_ids, 1] = _normalize(
            items_df["book_ratings_count"].values, norm_stats["book_ratings_count"]
        )
        num_arr[item_ids, 2] = _normalize(
            items_df["book_num_pages"].values, norm_stats["book_num_pages"]
        )
        pub_year = items_df["book_publication_year"].fillna(pub_year_median).values
        num_arr[item_ids, 3] = _normalize(
            pub_year, norm_stats["book_publication_year"], fill_null_with=pub_year_median
        )
        num_arr[item_ids, 4] = items_df["book_is_ebook"].fillna(0).values.astype(np.float32)

    return torch.from_numpy(cat_arr), torch.from_numpy(num_arr)


def _normalize(
    values: np.ndarray,
    stats: dict[str, Any],
    fill_null_with: float = 0.0,
) -> np.ndarray:
    """
    Apply log1p (if stats["use_log1p"]) then min-max normalization.
    NaN values are filled with fill_null_with before normalization.
    Result is clipped to [0, 1].
    """
    arr = np.where(np.isnan(values.astype(float)), fill_null_with, values.astype(float))
    if stats["use_log1p"]:
        arr = np.log1p(arr)
    lo, hi = stats["min"], stats["max"]
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


# ── Dataset ────────────────────────────────────────────────────────────────────

class TwoTowerDataset(Dataset):
    """
    PyTorch Dataset for two-tower training/validation.

    Each sample corresponds to one positive user-item interaction row from the
    Parquet split. Item features are not stored per-row — they are looked up
    from the pre-built item_cat_feats / item_num_feats tensors at __getitem__
    time using target_item_id as the index (O(1) tensor index, very fast).

    Parameters
    ----------
    df : pd.DataFrame
        The split dataframe (train or val).
    item_cat_feats : LongTensor [num_items+1, 6]
    item_num_feats : FloatTensor [num_items+1, 5]
    cfg : DictConfig
        Full config (uses cfg.history.length).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        item_cat_feats: Tensor,
        item_num_feats: Tensor,
        cfg: DictConfig,
    ) -> None:
        self.item_cat_feats = item_cat_feats
        self.item_num_feats = item_num_feats
        self.history_length = cfg.history.length

        # Store the columns we need as numpy arrays for fast __getitem__
        self.user_ids = df["user_id"].values.astype(np.int32)
        self.target_item_ids = df["target_item_id"].values.astype(np.int32)
        self.sample_weights = df["sample_weight"].values.astype(np.float32)

        # history_item_ids and history_item_weights are stored as arrays of lists;
        # convert to 2D numpy arrays for direct indexing.
        # tolist() avoids the intermediate Python list-of-arrays that np.stack+apply creates,
        # cutting peak memory roughly in half for these columns.
        self.history_item_ids = np.array(df["history_item_ids"].tolist(), dtype=np.int32)    # [N, 10]
        self.history_item_weights = np.array(df["history_item_weights"].tolist(), dtype=np.float32)  # [N, 10]

    @classmethod
    def from_parquet(
        cls,
        path: str,
        columns: list[str],
        item_cat_feats: Tensor,
        item_num_feats: Tensor,
        cfg: DictConfig,
        mmap_dir: Path | None = None,
        split_name: str = "split",
    ) -> "TwoTowerDataset":
        """
        Build a TwoTowerDataset by reading the Parquet in batches, concatenating
        only the lightweight scalar/array columns into numpy arrays — never holding
        the full DataFrame in memory at once.

        If mmap_dir is given, arrays are saved as .npy files on first run and
        loaded with mmap_mode='r' on subsequent runs, so only accessed pages
        are kept in RAM.
        """
        if mmap_dir is not None:
            mmap_dir = Path(mmap_dir) / split_name
            files = {
                "user_ids":             mmap_dir / "user_ids.npy",
                "target_item_ids":      mmap_dir / "target_item_ids.npy",
                "sample_weights":       mmap_dir / "sample_weights.npy",
                "history_item_ids":     mmap_dir / "history_item_ids.npy",
                "history_item_weights": mmap_dir / "history_item_weights.npy",
            }
            if all(f.exists() for f in files.values()):
                print(f"  Loading memory-mapped arrays from {mmap_dir}")
                obj = cls.__new__(cls)
                obj.item_cat_feats = item_cat_feats
                obj.item_num_feats = item_num_feats
                obj.history_length = cfg.history.length
                obj.user_ids             = np.load(files["user_ids"],             mmap_mode="r")
                obj.target_item_ids      = np.load(files["target_item_ids"],      mmap_mode="r")
                obj.sample_weights       = np.load(files["sample_weights"],       mmap_mode="r")
                obj.history_item_ids     = np.load(files["history_item_ids"],     mmap_mode="r")
                obj.history_item_weights = np.load(files["history_item_weights"], mmap_mode="r")
                return obj
            mmap_dir.mkdir(parents=True, exist_ok=True)

        user_ids_list: list[np.ndarray] = []
        target_item_ids_list: list[np.ndarray] = []
        sample_weights_list: list[np.ndarray] = []
        history_item_ids_list: list[np.ndarray] = []
        history_item_weights_list: list[np.ndarray] = []

        for batch in _iter_parquet_batches(path, columns=columns):
            user_ids_list.append(batch["user_id"].values.astype(np.int32))
            target_item_ids_list.append(batch["target_item_id"].values.astype(np.int32))
            sample_weights_list.append(batch["sample_weight"].values.astype(np.float32))
            history_item_ids_list.append(
                np.array(batch["history_item_ids"].tolist(), dtype=np.int32)
            )
            history_item_weights_list.append(
                np.array(batch["history_item_weights"].tolist(), dtype=np.float32)
            )

        obj = cls.__new__(cls)
        obj.item_cat_feats = item_cat_feats
        obj.item_num_feats = item_num_feats
        obj.history_length = cfg.history.length
        obj.user_ids             = np.concatenate(user_ids_list)
        obj.target_item_ids      = np.concatenate(target_item_ids_list)
        obj.sample_weights       = np.concatenate(sample_weights_list)
        obj.history_item_ids     = np.concatenate(history_item_ids_list)      # [N, 10]
        obj.history_item_weights = np.concatenate(history_item_weights_list)  # [N, 10]

        if mmap_dir is not None:
            print(f"  Saving memory-mapped arrays to {mmap_dir}")
            np.save(files["user_ids"],             obj.user_ids)
            np.save(files["target_item_ids"],      obj.target_item_ids)
            np.save(files["sample_weights"],       obj.sample_weights)
            np.save(files["history_item_ids"],     obj.history_item_ids)
            np.save(files["history_item_weights"], obj.history_item_weights)
            # Replace in-memory arrays with mmap'd versions to free RAM now
            obj.user_ids             = np.load(files["user_ids"],             mmap_mode="r")
            obj.target_item_ids      = np.load(files["target_item_ids"],      mmap_mode="r")
            obj.sample_weights       = np.load(files["sample_weights"],       mmap_mode="r")
            obj.history_item_ids     = np.load(files["history_item_ids"],     mmap_mode="r")
            obj.history_item_weights = np.load(files["history_item_weights"], mmap_mode="r")

        return obj

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        item_id = int(self.target_item_ids[idx])
        return {
            "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "target_item_id": torch.tensor(item_id, dtype=torch.long),
            "history_item_ids": torch.from_numpy(self.history_item_ids[idx]).long(),
            "history_item_weights": torch.from_numpy(self.history_item_weights[idx]),
            "sample_weight": torch.tensor(self.sample_weights[idx], dtype=torch.float32),
            "target_cat_feats": self.item_cat_feats[item_id],   # LongTensor [6]
            "target_num_feats": self.item_num_feats[item_id],   # FloatTensor [5]
        }


# ── Top-level entry point ──────────────────────────────────────────────────────

def prepare_data(cfg: DictConfig) -> tuple[
    TwoTowerDataset,
    TwoTowerDataset,
    Tensor,
    Tensor,
    dict[str, Any],
]:
    """
    Master entry point for data preparation.

    On first run: reads training Parquet, builds vocabs + norm stats, saves to
    cfg.data.artifacts_path, builds item feature tensors, wraps datasets.

    On subsequent runs: loads artifacts from cache, skips vocab/stat computation.

    Parameters
    ----------
    cfg : DictConfig
        Full config loaded from baseline.yaml.

    Returns
    -------
    train_dataset  : TwoTowerDataset
    val_dataset    : TwoTowerDataset
    item_cat_feats : LongTensor  [num_items+1, 6]  — shared, in shared memory
    item_num_feats : FloatTensor [num_items+1, 5]  — shared, in shared memory
    artifacts      : dict  — vocabs + norm stats (use to set cfg.vocab.* before model init)
    """
    artifacts_path = Path(cfg.data.artifacts_path)

    # Columns needed only for building vocab/norm artifacts
    _ARTIFACT_COLS = [
        "user_id", "target_item_id",
        "book_primary_author_id", "book_language", "book_format", "book_top_shelves",
        "book_avg_rating", "book_ratings_count", "book_num_pages",
        "book_publication_year", "book_is_ebook",
    ]
    # Columns needed only for building the item feature tensors
    _ITEM_FEAT_COLS = [
        "target_item_id",
        "book_primary_author_id", "book_language", "book_format", "book_top_shelves",
        "book_avg_rating", "book_ratings_count", "book_num_pages",
        "book_publication_year", "book_is_ebook",
    ]
    # Columns needed for TwoTowerDataset
    _DATASET_COLS = [
        "user_id", "target_item_id", "sample_weight",
        "history_item_ids", "history_item_weights",
    ]

    if artifacts_path.exists():
        print(f"Loading artifacts from {artifacts_path}")
        artifacts = load_artifacts(artifacts_path)
    else:
        print("Building artifacts from training split (chunked)...")
        artifacts = build_artifacts(
            _iter_parquet_batches(cfg.data.train_path, columns=_ARTIFACT_COLS),
            cfg,
        )
        save_artifacts(artifacts, artifacts_path)
        print(f"Artifacts saved to {artifacts_path}")

    print(f"  num_users={artifacts['num_users']:,}  num_items={artifacts['num_items']:,}")
    print(f"  author vocab size: {len(artifacts['author_vocab']):,}")
    print(f"  language vocab size: {len(artifacts['language_vocab']):,}")
    print(f"  format vocab size: {len(artifacts['format_vocab']):,}")
    print(f"  shelf vocab size: {len(artifacts['shelf_vocab']):,}")

    print("Building item feature tensors (chunked)...")
    item_cat_feats, item_num_feats = build_item_feature_tensors(
        _iter_parquet_batches(cfg.data.train_path, columns=_ITEM_FEAT_COLS),
        artifacts,
    )
    gc.collect()

    # Share memory so DataLoader workers don't copy these tensors
    item_cat_feats.share_memory_()
    item_num_feats.share_memory_()

    print(f"  item_cat_feats: {tuple(item_cat_feats.shape)}  {item_cat_feats.dtype}")
    print(f"  item_num_feats: {tuple(item_num_feats.shape)}  {item_num_feats.dtype}")

    mmap_dir = Path(cfg.data.mmap_dir) if cfg.data.get("mmap_dir") else None

    print("Wrapping train dataset...")
    train_dataset = TwoTowerDataset.from_parquet(
        cfg.data.train_path, _DATASET_COLS, item_cat_feats, item_num_feats, cfg,
        mmap_dir=mmap_dir, split_name="train",
    )
    gc.collect()

    print("Wrapping val dataset...")
    val_dataset = TwoTowerDataset.from_parquet(
        cfg.data.val_path, _DATASET_COLS, item_cat_feats, item_num_feats, cfg,
        mmap_dir=mmap_dir, split_name="val",
    )
    gc.collect()

    print(f"  train: {len(train_dataset):,} rows  |  val: {len(val_dataset):,} rows")

    return train_dataset, val_dataset, item_cat_feats, item_num_feats, artifacts
