"""
Two-Tower Data Preparation — Full Pipeline (Dataproc)

Transforms raw Goodreads interactions into PyTorch-ready Parquet files:
  - Maps hashed user/book IDs to integer IDs
  - Computes interaction strength
  - Deduplicates per (user, item) pair
  - Iterative cold-start filtering
  - Per-user temporal split (70/20/10)
  - Builds fixed-length user history arrays (padded)
  - Attaches book metadata features
  - Writes train/val/test splits to GCS

Output schema:
  user_id                int
  target_item_id         int
  history_item_ids       array<int>[10]     (padded with 0)
  history_item_weights   array<float>[10]   (interaction strengths, padded with 0.0)
  sample_weight          float
  timestamp              timestamp
  is_read                int   (0/1)
  rating                 int   (0-5)
  book_avg_rating        float
  book_ratings_count     int
  book_language          string
  book_format            string
  book_num_pages         int
  book_publication_year  int
  book_is_ebook          int   (0/1)
  book_primary_author_id string
  book_top_shelves       array<string>[3]
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, IntegerType, FloatType, StringType


# ---------------------------------------------------------------------------
# Logging — tee to stdout AND a timestamped log file
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> str:
    """Configure root logger to write to both stdout and a timestamped log file.

    Returns the resolved path to the log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"01_data_prep_{ts}.log")

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return log_path


def log(msg: str = "") -> None:
    logging.info(msg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Two-tower data preparation")
    p.add_argument(
        "--gcs-base",
        default="gs://nshen7-personal-bucket/projects/rec_sys_goodreads",
        help="GCS base path (no trailing slash)",
    )
    p.add_argument(
        "--output-path",
        default="gs://nshen7-personal-bucket/projects/rec_sys_goodreads/data/two_tower_splits",
        help="GCS output path for train/val/test parquet",
    )
    p.add_argument("--min-user-interactions", type=int, default=5)
    p.add_argument("--min-item-interactions", type=int, default=5)
    p.add_argument("--beta", type=float, default=1.0,
                   help="Weight for positive ratings in interaction strength formula")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--history-length", type=int, default=10)
    p.add_argument("--top-shelves", type=int, default=3)
    p.add_argument("--checkpoint-dir", default="/tmp/spark_checkpoints")
    p.add_argument(
        "--sample-pct", type=float, default=20.0,
        help="Percentage of users to sample (1-100). 100 = use all users.",
    )
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument(
        "--log-dir",
        default="gs://nshen7-personal-bucket/projects/rec_sys_goodreads/two_tower/jobs/logs",
        help="Directory where the timestamped log file will be written",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------

def create_spark(app_name: str = "two_tower_data_prep") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def load_data(spark, gcs_base: str):
    interactions = spark.read.parquet(f"{gcs_base}/data/parquet/goodreads_interactions_dedup")
    user_id_map  = spark.read.parquet(f"{gcs_base}/data/parquet/user_id_map")
    book_id_map  = spark.read.parquet(f"{gcs_base}/data/parquet/book_id_map")
    log(f"Loaded interactions: {interactions.count():,}")
    log(f"User ID map:         {user_id_map.count():,}")
    log(f"Book ID map:         {book_id_map.count():,}")
    return interactions, user_id_map, book_id_map


def sample_users(interactions, sample_pct: float, seed: int):
    """User-level Bernoulli sampling: sample sample_pct% of users, keep all their interactions."""
    if sample_pct >= 100.0:
        log("sample_pct=100 — using all users, skipping sampling")
        return interactions

    fraction = sample_pct / 100.0
    all_users = interactions.select("user_id").distinct()
    total_users = all_users.count()
    sampled_users = all_users.sample(fraction=fraction, seed=seed)
    sampled_count = sampled_users.count()
    log(f"Sampled {sampled_count:,} / {total_users:,} users ({sampled_count / total_users * 100:.1f}%)")

    interactions = interactions.join(F.broadcast(sampled_users), "user_id", "inner")
    log(f"Interactions after sampling: {interactions.count():,}")
    return interactions


def map_ids(interactions, user_id_map, book_id_map):
    """Join hashed string IDs -> integer IDs and select core columns."""
    df = (
        interactions
        .join(F.broadcast(user_id_map),
              interactions["user_id"] == user_id_map["user_id"],
              "inner")
        .drop(user_id_map["user_id"])
    )
    df = (
        df
        .withColumn("book_id_int", F.col("book_id").cast("int"))
        .join(F.broadcast(book_id_map),
              F.col("book_id_int") == book_id_map["book_id"],
              "inner")
        .drop(book_id_map["book_id"])
    )
    df = df.select(
        F.col("user_id_csv").alias("user_id"),
        F.col("book_id_csv").alias("item_id"),
        F.col("is_read").cast("int").alias("is_read"),
        F.col("rating").cast("int").alias("rating"),
        F.col("date_added"),
    )
    log(f"After ID mapping: {df.count():,} rows")
    return df


def build_book_features(spark, gcs_base: str, book_id_map, top_shelves: int):
    """Load books parquet and extract/clean metadata features."""
    books_raw = spark.read.parquet(f"{gcs_base}/data/parquet/books")

    books_features = (
        books_raw
        .withColumn(
            "book_avg_rating",
            F.nullif(F.trim(F.col("average_rating")), F.lit("")).cast("float"),
        )
        .withColumn(
            "book_ratings_count",
            F.nullif(F.trim(F.col("ratings_count")), F.lit("")).cast("int"),
        )
        .withColumn(
            "book_language",
            F.when(
                F.col("language_code").isNull() | (F.trim(F.col("language_code")) == ""),
                F.lit("unknown"),
            ).otherwise(F.col("language_code")),
        )
        .withColumn(
            "book_format",
            F.when(
                F.col("format").isNull() | (F.trim(F.col("format")) == ""),
                F.lit("unknown"),
            ).otherwise(F.col("format")),
        )
        .withColumn(
            "book_num_pages",
            F.nullif(F.trim(F.col("num_pages")), F.lit("")).cast("int"),
        )
        .withColumn(
            "book_publication_year",
            F.nullif(F.trim(F.col("publication_year")), F.lit("")).cast("int"),
        )
        .withColumn(
            "book_is_ebook",
            F.when(F.lower(F.col("is_ebook")) == "true", F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "book_primary_author_id",
            F.get(F.col("authors"), 0)["author_id"].cast("string"),
        )
        .withColumn(
            "_shelf_names",
            F.transform(
                F.slice(F.coalesce(F.col("popular_shelves"), F.array()), 1, top_shelves),
                lambda s: s["name"],
            ),
        )
        .withColumn(
            "book_top_shelves",
            F.concat(
                F.col("_shelf_names"),
                F.array_repeat(
                    F.lit(""),
                    F.greatest(F.lit(0), F.lit(top_shelves) - F.size(F.col("_shelf_names"))),
                ),
            ),
        )
        .select(
            F.col("book_id").alias("_book_id_str"),
            "book_avg_rating",
            "book_ratings_count",
            "book_language",
            "book_format",
            "book_num_pages",
            "book_publication_year",
            "book_is_ebook",
            "book_primary_author_id",
            "book_top_shelves",
        )
    )

    book_features = (
        book_id_map
        .join(books_features,
              book_id_map["book_id"] == books_features["_book_id_str"],
              "inner")
        .select(
            F.col("book_id_csv").alias("item_id"),
            "book_avg_rating",
            "book_ratings_count",
            "book_language",
            "book_format",
            "book_num_pages",
            "book_publication_year",
            "book_is_ebook",
            "book_primary_author_id",
            "book_top_shelves",
        )
    )
    log(f"Book features processed: {book_features.count():,} books")
    return book_features


def parse_timestamps(df):
    """Parse date_added string -> timestamp; drop rows that fail."""
    df = df.withColumn(
        "date_added_ts",
        F.try_to_timestamp(F.col("date_added"), F.lit("EEE MMM dd HH:mm:ss Z yyyy")),
    )
    n_total = df.count()
    n_parsed = df.filter(F.col("date_added_ts").isNotNull()).count()
    n_failed = n_total - n_parsed
    log(f"Timestamp parsing: {n_parsed:,} OK ({n_parsed/n_total*100:.1f}%), "
        f"{n_failed:,} failed ({n_failed/n_total*100:.1f}%)")

    if n_failed > n_total * 0.5:
        log("Primary format failed >50% — trying unix_timestamp fallback...")
        df = df.drop("date_added_ts").withColumn(
            "date_added_ts",
            F.from_unixtime(
                F.unix_timestamp(F.col("date_added"), "EEE MMM dd HH:mm:ss Z yyyy")
            ).cast("timestamp"),
        )

    df = df.filter(F.col("date_added_ts").isNotNull())
    n_valid = df.count()
    log(f"Rows with valid timestamps: {n_valid:,}")

    ts_row = df.select(
        F.min("date_added_ts").alias("earliest"),
        F.max("date_added_ts").alias("latest"),
    ).collect()[0]
    log(f"Timestamp range: {ts_row['earliest']}  ->  {ts_row['latest']}")

    return df


def compute_interaction_strength(df, beta: float):
    df = df.withColumn(
        "interaction_strength",
        F.lit(1.0)
        + F.lit(2.0) * F.col("is_read")
        + F.lit(beta) * F.greatest(F.lit(0), F.col("rating") - F.lit(3)),
    )
    log(f"Interaction strength formula: r = 1 + 2*is_read + {beta}*max(0, rating-3)")
    dist_rows = (
        df.groupBy("interaction_strength")
        .count()
        .orderBy("interaction_strength")
        .collect()
    )
    log("Interaction strength distribution:")
    for r in dist_rows:
        log(f"  strength={r['interaction_strength']:.1f}  count={r['count']:,}")
    return df


def split_confirmed_negatives(df):
    """Separate is_read=1, rating∈{1,2} rows into a confirmed-negative table.

    Returns (positives_df, confirmed_negatives_df).
    confirmed_negatives_df has columns: user_id, item_id, rating.
    """
    neg_mask = (F.col("is_read") == 1) & (F.col("rating").isin(1, 2))
    confirmed_negatives = df.filter(neg_mask).select("user_id", "item_id", "rating")
    positives = df.filter(~neg_mask)
    n_neg = confirmed_negatives.count()
    n_pos = positives.count()
    log(f"Confirmed negatives (is_read=1, rating∈{{1,2}}): {n_neg:,}")
    log(f"Positives remaining:                             {n_pos:,}")
    return positives, confirmed_negatives


def deduplicate(df):
    """Keep the most recent interaction per (user, item) pair."""
    n_before = df.count()
    window_dedup = Window.partitionBy("user_id", "item_id").orderBy(F.desc("date_added_ts"))
    df = (
        df.withColumn("rn", F.row_number().over(window_dedup))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )
    n_after = df.count()
    log(f"Deduplication: {n_before:,} -> {n_after:,} (removed {n_before - n_after:,}, "
        f"{(n_before - n_after) / n_before * 100:.2f}%)")
    return df


def cold_start_filter(spark, df, min_user: int, min_item: int, checkpoint_dir: str):
    """Iteratively remove users/items with fewer than min interactions."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    df = df.select("user_id", "item_id", "is_read", "rating",
                   "date_added_ts", "interaction_strength")

    prev_count = 0
    curr_count = df.count()
    iteration = 0
    log(f"Cold-start filtering start: {curr_count:,} interactions")

    while curr_count != prev_count:
        iteration += 1
        prev_count = curr_count

        user_counts = (
            df.groupBy("user_id").agg(F.count("*").alias("n"))
            .filter(F.col("n") >= min_user).select("user_id")
        )
        df = df.join(user_counts, "user_id", "inner")

        item_counts = (
            df.groupBy("item_id").agg(F.count("*").alias("n"))
            .filter(F.col("n") >= min_item).select("item_id")
        )
        df = df.join(item_counts, "item_id", "inner")

        df = df.checkpoint()
        df = df.cache()
        curr_count = df.count()
        n_users = user_counts.count()
        n_items = item_counts.count()
        log(f"  Iter {iteration}: {curr_count:,} interactions, "
            f"{n_users:,} users, {n_items:,} items")

    log(f"Converged after {iteration} iterations. Final: {curr_count:,} interactions")
    surviving_users = df.select("user_id").distinct()
    surviving_items = df.select("item_id").distinct()
    return df, surviving_users, surviving_items


def temporal_split(df, train_ratio: float, val_ratio: float):
    """Per-user temporal split. Returns df_indexed with 'split' column."""
    window_user = Window.partitionBy("user_id").orderBy("date_added_ts")

    df_indexed = (
        df
        .withColumn("user_idx", F.row_number().over(window_user))
        .withColumn("user_count", F.count("*").over(Window.partitionBy("user_id")))
        .withColumn(
            "split",
            F.when(F.col("user_idx") <= F.col("user_count") * train_ratio, F.lit("train"))
            .when(
                F.col("user_idx") <= F.col("user_count") * (train_ratio + val_ratio),
                F.lit("val"),
            )
            .otherwise(F.lit("test")),
        )
    )
    df_indexed = df_indexed.cache()
    df_indexed.count()

    dist_rows = df_indexed.groupBy("split").count().orderBy("split").collect()
    log("Raw split distribution (before post-split filtering):")
    for r in dist_rows:
        log(f"  {r['split']:5s}: {r['count']:,}")

    # Post-split filtering: val/test must only contain users/items seen in train
    train_users = df_indexed.filter(F.col("split") == "train").select("user_id").distinct()
    train_items = df_indexed.filter(F.col("split") == "train").select("item_id").distinct()

    val = (
        df_indexed.filter(F.col("split") == "val")
        .join(F.broadcast(train_users), "user_id", "inner")
        .join(F.broadcast(train_items), "item_id", "inner")
    )
    test = (
        df_indexed.filter(F.col("split") == "test")
        .join(F.broadcast(train_users), "user_id", "inner")
        .join(F.broadcast(train_items), "item_id", "inner")
    )
    train = df_indexed.filter(F.col("split") == "train")

    n_train = train.count()
    n_val   = val.count()
    n_test  = test.count()
    n_total = n_train + n_val + n_test
    log("Final split counts (after post-split user/item filtering):")
    log(f"  train: {n_train:,} ({n_train/n_total*100:.1f}%)")
    log(f"  val:   {n_val:,}   ({n_val/n_total*100:.1f}%)")
    log(f"  test:  {n_test:,}  ({n_test/n_total*100:.1f}%)")

    # Re-attach split label so we can work from a single df_indexed
    # (val/test had users/items filtered; rebuild df_indexed with filtered tags)
    val = val.withColumn("split", F.lit("val"))
    test = test.withColumn("split", F.lit("test"))
    df_indexed = train.union(val).union(test).cache()
    df_indexed.count()

    return df_indexed


def build_user_histories(df_indexed, history_length: int, pad_item_id: int = 0):
    """
    For each row, collect the item_ids / interaction_strengths of all
    *prior* interactions (within the same user), take the last history_length,
    and pad to a fixed-length array.
    """
    window_hist = (
        Window.partitionBy("user_id")
        .orderBy("date_added_ts")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    # Collect all previous item_ids and strengths as arrays
    df_indexed = df_indexed.withColumn(
        "_raw_hist_ids",
        F.collect_list(F.col("item_id")).over(window_hist),
    ).withColumn(
        "_raw_hist_weights",
        F.collect_list(F.col("interaction_strength")).over(window_hist),
    )

    # Slice to last `history_length` entries (most recent)
    df_indexed = df_indexed.withColumn(
        "_hist_len", F.size(F.col("_raw_hist_ids"))
    ).withColumn(
        "_slice_start",
        F.greatest(F.lit(1), F.col("_hist_len") - F.lit(history_length) + F.lit(1)),
    ).withColumn(
        "_hist_ids_trimmed",
        F.slice(F.col("_raw_hist_ids"), F.col("_slice_start"), F.lit(history_length)),
    ).withColumn(
        "_hist_weights_trimmed",
        F.slice(F.col("_raw_hist_weights"), F.col("_slice_start"), F.lit(history_length)),
    )

    # Pad to exactly history_length
    df_indexed = df_indexed.withColumn(
        "_pad_len",
        F.greatest(F.lit(0), F.lit(history_length) - F.size(F.col("_hist_ids_trimmed"))),
    ).withColumn(
        "history_item_ids",
        F.concat(
            F.col("_hist_ids_trimmed"),
            F.array_repeat(F.lit(pad_item_id), F.col("_pad_len")),
        ).cast(ArrayType(IntegerType())),
    ).withColumn(
        "history_item_weights",
        F.concat(
            F.col("_hist_weights_trimmed"),
            F.array_repeat(F.lit(0.0), F.col("_pad_len")),
        ).cast(ArrayType(FloatType())),
    )

    # Drop intermediate columns
    drop_cols = [
        "_raw_hist_ids", "_raw_hist_weights",
        "_hist_len", "_slice_start",
        "_hist_ids_trimmed", "_hist_weights_trimmed",
        "_pad_len",
    ]
    df_indexed = df_indexed.drop(*drop_cols)

    return df_indexed


def attach_book_features_and_finalise(df_indexed, book_features):
    """Left-join book features onto df_indexed and select final columns."""
    df = df_indexed.join(book_features, "item_id", "left")

    final_cols = [
        F.col("user_id"),
        F.col("item_id").alias("target_item_id"),
        F.col("history_item_ids"),
        F.col("history_item_weights"),
        F.col("interaction_strength").alias("sample_weight"),
        F.col("date_added_ts").alias("timestamp"),
        F.col("is_read"),
        F.col("rating"),
        "book_avg_rating",
        "book_ratings_count",
        "book_language",
        "book_format",
        "book_num_pages",
        "book_publication_year",
        "book_is_ebook",
        "book_primary_author_id",
        "book_top_shelves",
        "split",
    ]
    return df.select(*final_cols)


def validate_final(df_final, history_length: int) -> None:
    """Run the same sanity checks as the notebook's validation section."""
    log("")
    log("=" * 55)
    log("VALIDATION CHECKS")
    log("=" * 55)

    train = df_final.filter(F.col("split") == "train")
    val   = df_final.filter(F.col("split") == "val")
    test  = df_final.filter(F.col("split") == "test")

    # [1] No nulls in critical columns
    log("[1/5] Null counts in critical columns (train):")
    critical = ["user_id", "target_item_id", "history_item_ids", "history_item_weights",
                "sample_weight", "timestamp"]
    null_row = train.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in critical]
    ).collect()[0]
    all_ok = True
    for c in critical:
        count = null_row[c]
        flag = "OK" if count == 0 else "FAIL"
        log(f"  {c}: {count} nulls  [{flag}]")
        if count > 0:
            all_ok = False
    if all_ok:
        log("  => No nulls in critical columns")

    # [2] History array lengths
    log(f"[2/5] History array lengths (must all be {history_length}):")
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        bad = split_df.filter(F.size(F.col("history_item_ids")) != history_length).count()
        flag = "OK" if bad == 0 else f"FAIL — {bad:,} rows with wrong length"
        log(f"  {name}: {flag}")

    # [3] Temporal ordering note
    log("[3/5] Temporal ordering (global min/max per split):")
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        row = split_df.select(
            F.min("timestamp").alias("mn"), F.max("timestamp").alias("mx")
        ).collect()[0]
        log(f"  {name}: {row['mn']}  ->  {row['mx']}")
    log("  Note: global overlap is expected — ordering is enforced per-user")

    # [4] Sample weight distribution
    log("[4/5] Sample weight distribution (train):")
    w = train.select(
        F.min("sample_weight").alias("min"),
        F.avg("sample_weight").alias("avg"),
        F.max("sample_weight").alias("max"),
    ).collect()[0]
    log(f"  min={w['min']:.1f}  avg={w['avg']:.2f}  max={w['max']:.1f}")
    dist_w = (
        train.groupBy("sample_weight").count()
        .orderBy("sample_weight").collect()
    )
    for r in dist_w:
        log(f"  weight={r['sample_weight']:.1f}  count={r['count']:,}")

    # [5] Book feature coverage
    log("[5/5] Book feature coverage (train):")
    n_train = train.count()
    n_missing = train.filter(F.col("book_avg_rating").isNull()).count()
    log(f"  With book features:    {n_train - n_missing:,} ({(n_train - n_missing)/n_train*100:.1f}%)")
    log(f"  Missing book features: {n_missing:,} ({n_missing/n_train*100:.1f}%)")

    log("")
    log("User/item coverage per split:")
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        n_users = split_df.select("user_id").distinct().count()
        n_items = split_df.select("target_item_id").distinct().count()
        log(f"  {name}: {n_users:,} unique users, {n_items:,} unique items")

    log("=" * 55)
    log("VALIDATION COMPLETE")
    log("=" * 55)


def write_splits(df_final, output_path: str):
    for split in ("train", "val", "test"):
        path = f"{output_path}/{split}"
        split_df = df_final.filter(F.col("split") == split).drop("split")
        split_df.write.mode("overwrite").parquet(path)
        n = split_df.count()
        log(f"Wrote {split}: {n:,} rows -> {path}")


def write_confirmed_negatives(confirmed_negatives, surviving_users, surviving_items, output_path: str):
    """Filter confirmed negatives to training users/items and write to GCS."""
    path = f"{output_path}/confirmed_negatives"
    df = (
        confirmed_negatives
        .join(F.broadcast(surviving_users), "user_id", "inner")
        .join(F.broadcast(surviving_items), "item_id", "inner")
    )
    df.write.mode("overwrite").parquet(path)
    n = df.count()
    log(f"Wrote confirmed_negatives: {n:,} rows -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    log_path = setup_logging(args.log_dir)

    spark = create_spark()

    log("=" * 60)
    log("TWO-TOWER DATA PREPARATION — START")
    log("=" * 60)
    log(f"Log file: {log_path}")
    log(f"GCS base: {args.gcs_base}")
    log(f"Output:   {args.output_path}")
    log(f"Sample pct: {args.sample_pct}%  seed={args.sample_seed}")
    log(f"Min interactions: user>={args.min_user_interactions}, item>={args.min_item_interactions}")
    log(f"History length: {args.history_length}  |  Top shelves: {args.top_shelves}")
    log(f"Beta: {args.beta}  |  Split: {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio:.1f}")
    log("")

    log("=== Step 1: Load data ===")
    interactions, user_id_map, book_id_map = load_data(spark, args.gcs_base)

    log("\n=== Step 1b: Sample users ===")
    interactions = sample_users(interactions, args.sample_pct, args.sample_seed)

    log("\n=== Step 2: Map IDs ===")
    df = map_ids(interactions, user_id_map, book_id_map)

    log("\n=== Step 3: Build book features ===")
    book_features = build_book_features(spark, args.gcs_base, book_id_map, args.top_shelves)

    log("\n=== Step 4: Parse timestamps ===")
    df = parse_timestamps(df)

    log("\n=== Step 5: Compute interaction strength ===")
    df = compute_interaction_strength(df, args.beta)

    log("\n=== Step 5b: Split confirmed negatives ===")
    df, confirmed_negatives = split_confirmed_negatives(df)

    log("\n=== Step 6: Deduplicate ===")
    df = deduplicate(df)

    log("\n=== Step 7: Cold-start filtering ===")
    df, surviving_users, surviving_items = cold_start_filter(
        spark, df,
        args.min_user_interactions,
        args.min_item_interactions,
        args.checkpoint_dir,
    )

    log("\n=== Step 8: Temporal split ===")
    df_indexed = temporal_split(df, args.train_ratio, args.val_ratio)

    log("\n=== Step 9: Build user histories ===")
    df_indexed = build_user_histories(df_indexed, args.history_length)

    log("\n=== Step 10: Attach book features & finalise schema ===")
    df_final = attach_book_features_and_finalise(df_indexed, book_features)

    log("\n=== Step 11: Validation checks ===")
    validate_final(df_final, args.history_length)

    log("\n=== Step 12: Write output ===")
    write_splits(df_final, args.output_path)
    write_confirmed_negatives(confirmed_negatives, surviving_users, surviving_items, args.output_path)

    log("")
    log("=" * 60)
    log("TWO-TOWER DATA PREPARATION — DONE")
    log(f"Log written to: {log_path}")
    log("=" * 60)
    spark.stop()


if __name__ == "__main__":
    main()


