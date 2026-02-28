"""
02 — Temporal Train / Validation / Test Split (Dataproc version)

Prepares data for PySpark implicit ALS by keeping ALL interactions
(shelved, read, rated) and outputting (user_id, book_id, is_read, rating).

Submits via:
    gcloud dataproc batches submit pyspark \
        gs://<BUCKET>/projects/rec_sys_goodreads/scripts/02_data_split.py \
        --region=us-central1 \
        --properties="spark.driver.memory=4g,spark.executor.memory=4g,spark.sql.legacy.timeParserPolicy=LEGACY"
"""

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F

# ===================================================================
# Configuration
# ===================================================================
SAMPLE_PCT = 20.0  # Percentage of users to randomly sample (all their interactions are kept)
MIN_USER_INTERACTIONS = 5
MIN_BOOK_INTERACTIONS = 5

# --- Spark session (Dataproc handles GCS connector, Java, auth) ---
spark = (
    SparkSession.builder
    .appName("goodreads_data_split")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .getOrCreate()
)

GCS_BASE = "gs://nshen7-personal-bucket/projects/rec_sys_goodreads"
PARQUET_BASE = f"{GCS_BASE}/data/parquet"
OUTPUT_BASE = f"{GCS_BASE}/data/splits_sample_{int(SAMPLE_PCT)}pct"

print(f"Configuration: Using {SAMPLE_PCT}% of users (all their interactions kept)")
print(f"Output base: {OUTPUT_BASE}")

# ===================================================================
# 1. Load data — ALL interactions (every row = a shelved event)
# ===================================================================
interactions_dedup = spark.read.parquet(f"{PARQUET_BASE}/goodreads_interactions_dedup")
user_id_map = spark.read.parquet(f"{PARQUET_BASE}/user_id_map")
book_id_map = spark.read.parquet(f"{PARQUET_BASE}/book_id_map")

print(f"interactions_dedup (all): {interactions_dedup.count():,} rows")
print(f"user_id_map:              {user_id_map.count():,} rows")
print(f"book_id_map:              {book_id_map.count():,} rows")

# ===================================================================
# 2. Join with ID maps (broadcast small tables)
# ===================================================================
df = (
    interactions_dedup
    .join(
        F.broadcast(user_id_map),
        interactions_dedup["user_id"] == user_id_map["user_id"],
        "inner",
    )
    .drop(user_id_map["user_id"])
)

df = (
    df
    .withColumn("book_id_int", F.col("book_id").cast("int"))
    .join(
        F.broadcast(book_id_map),
        F.col("book_id_int") == book_id_map["book_id"],
        "inner",
    )
    .drop(book_id_map["book_id"])
)

df = df.select(
    F.col("user_id_csv").alias("user_id"),
    F.col("book_id_csv").alias("book_id"),
    F.col("is_read").cast("int").alias("is_read"),   # boolean -> 0/1
    F.col("rating").cast("int").alias("rating"),       # long -> int (0-5, 0=unrated)
    F.col("date_added"),
)

print(f"Joined dataset: {df.count():,} rows")

# ===================================================================
# 3. Parse timestamps
# ===================================================================
df = df.withColumn(
    "date_added_ts",
    F.to_timestamp(F.col("date_added"), "EEE MMM dd HH:mm:ss Z yyyy"),
)

n_parsed = df.filter(F.col("date_added_ts").isNotNull()).count()
n_failed = df.filter(F.col("date_added_ts").isNull()).count()
print(f"Timestamp parsed OK:    {n_parsed:,}")
print(f"Timestamp parse failed: {n_failed:,}")

if n_failed > n_parsed * 0.5:
    print("Primary format failed — trying unix_timestamp fallback...")
    df = df.drop("date_added_ts").withColumn(
        "date_added_ts",
        F.from_unixtime(
            F.unix_timestamp(F.col("date_added"), "EEE MMM dd HH:mm:ss Z yyyy")
        ).cast("timestamp"),
    )
    n_parsed = df.filter(F.col("date_added_ts").isNotNull()).count()
    n_failed = df.filter(F.col("date_added_ts").isNull()).count()
    print(f"Fallback parsed OK:    {n_parsed:,}")
    print(f"Fallback parse failed: {n_failed:,}")

df = df.filter(F.col("date_added_ts").isNotNull())

df.select(
    F.min("date_added_ts").alias("earliest"),
    F.max("date_added_ts").alias("latest"),
).show(truncate=False)

# ===================================================================
# 4. Sample users and keep all their interactions (user-level sampling)
# ===================================================================
if SAMPLE_PCT < 100.0:
    print(f"\nSampling {SAMPLE_PCT}% of users (keeping all their interactions)...")
    sample_fraction = SAMPLE_PCT / 100.0

    all_users = df.select("user_id").distinct()
    total_users = all_users.count()
    print(f"Total users before sampling: {total_users:,}")

    sampled_users = all_users.sample(fraction=sample_fraction, seed=42)
    sampled_user_count = sampled_users.count()
    print(f"Sampled users: {sampled_user_count:,} ({sampled_user_count/total_users*100:.1f}%)")

    df = df.join(F.broadcast(sampled_users), "user_id", "inner")
    sampled_count = df.count()
    print(f"Interactions after sampling: {sampled_count:,}")
else:
    print(f"\nUsing all interactions (SAMPLE_PCT = {SAMPLE_PCT}%)")

# ===================================================================
# 5. Iterative cold-start filtering
# ===================================================================
df_filtered = df.select("user_id", "book_id", "is_read", "rating", "date_added_ts")
spark.sparkContext.setCheckpointDir(f"{GCS_BASE}/data/_checkpoints")

prev_count = 0
curr_count = df_filtered.count()
iteration = 0

while curr_count != prev_count:
    iteration += 1
    prev_count = curr_count

    user_counts = (
        df_filtered.groupBy("user_id")
        .agg(F.count("*").alias("n"))
        .filter(F.col("n") >= MIN_USER_INTERACTIONS)
        .select("user_id")
    )
    df_filtered = df_filtered.join(user_counts, "user_id", "inner")

    book_counts = (
        df_filtered.groupBy("book_id")
        .agg(F.count("*").alias("n"))
        .filter(F.col("n") >= MIN_BOOK_INTERACTIONS)
        .select("book_id")
    )
    df_filtered = df_filtered.join(book_counts, "book_id", "inner")

    if iteration % 2 == 0:
        df_filtered = df_filtered.checkpoint()

    curr_count = df_filtered.count()
    n_users = df_filtered.select("user_id").distinct().count()
    n_books = df_filtered.select("book_id").distinct().count()
    print(
        f"Iteration {iteration}: {curr_count:,} interactions, "
        f"{n_users:,} users, {n_books:,} books"
    )

print(f"\nFinal filtered dataset: {curr_count:,} interactions")
df_filtered = df_filtered.cache()
df_filtered.count()

# ===================================================================
# 6. Per-user temporal split: train (70%) / val (20%) / test (10%)
#
# For each user, rank their interactions by date_added_ts.
# The last 10% (by rank) → test, the prior 20% → val, the rest → train.
# Every user appears in all three splits.
# ===================================================================
w = (
    Window
    .partitionBy("user_id")
    .orderBy(F.col("date_added_ts").asc())
)

df_ranked = df_filtered.withColumn(
    "rn", F.row_number().over(w)
).withColumn(
    "total", F.count("*").over(Window.partitionBy("user_id"))
)

# Compute per-row split boundaries
# test:  last 10%  → rn > total * 0.90
# val:   next 20%  → rn > total * 0.70 AND rn <= total * 0.90
# train: first 70% → rn <= total * 0.70
df_ranked = df_ranked.withColumn(
    "split",
    F.when(F.col("rn") > F.col("total") * 0.90, "test")
     .when(F.col("rn") > F.col("total") * 0.70, "val")
     .otherwise("train")
)

train = df_ranked.filter(F.col("split") == "train").drop("rn", "total", "split")
val   = df_ranked.filter(F.col("split") == "val"  ).drop("rn", "total", "split")
test  = df_ranked.filter(F.col("split") == "test" ).drop("rn", "total", "split")

n_train = train.count()
n_val   = val.count()
n_test  = test.count()
n_all   = n_train + n_val + n_test

print(f"Train: {n_train:,} ({n_train/n_all*100:.1f}%)")
print(f"Val:   {n_val:,} ({n_val/n_all*100:.1f}%)")
print(f"Test:  {n_test:,} ({n_test/n_all*100:.1f}%)")

# ===================================================================
# 7. Validation checks
# ===================================================================
# Every user in val/test must be in train (guaranteed by construction,
# but verify that no user ended up with zero train interactions).
train_users = train.select("user_id").distinct()
val_only_users   = val.select("user_id").distinct().subtract(train_users)
test_only_users  = test.select("user_id").distinct().subtract(train_users)

n_val_cold  = val_only_users.count()
n_test_cold = test_only_users.count()
assert n_val_cold  == 0, f"{n_val_cold} users in val have no train interactions!"
assert n_test_cold == 0, f"{n_test_cold} users in test have no train interactions!"
print("All val/test users have training history — no cold-start leakage.")

train_users_n = train.select("user_id").distinct().count()
train_books_n = train.select("book_id").distinct().count()
val_users_n   = val.select("user_id").distinct().count()
val_books_n   = val.select("book_id").distinct().count()
test_users_n  = test.select("user_id").distinct().count()
test_books_n  = test.select("book_id").distinct().count()

# ===================================================================
# 9. Save to GCS
# ===================================================================
train_als = train.select(
    F.col("user_id").cast("int"),
    F.col("book_id").cast("int"),
    F.col("is_read").cast("int"),
    F.col("rating").cast("int"),
)
val_als = val.select(
    F.col("user_id").cast("int"),
    F.col("book_id").cast("int"),
    F.col("is_read").cast("int"),
    F.col("rating").cast("int"),
)
test_als = test.select(
    F.col("user_id").cast("int"),
    F.col("book_id").cast("int"),
    F.col("is_read").cast("int"),
    F.col("rating").cast("int"),
)

train_als.write.mode("overwrite").parquet(f"{OUTPUT_BASE}/train")
val_als.write.mode("overwrite").parquet(f"{OUTPUT_BASE}/val")
test_als.write.mode("overwrite").parquet(f"{OUTPUT_BASE}/test")

# Verify
train_check = spark.read.parquet(f"{OUTPUT_BASE}/train").count()
val_check = spark.read.parquet(f"{OUTPUT_BASE}/val").count()
test_check = spark.read.parquet(f"{OUTPUT_BASE}/test").count()

print("=" * 60)
print("  DATA SPLIT SUMMARY")
print("=" * 60)
print(f"Sample percentage:      {SAMPLE_PCT}% of users")
print(f"Timestamp column:       date_added")
print(f"Split strategy:         per-user temporal (train 70% / val 20% / test 10%)")
print(f"Min user interactions:  {MIN_USER_INTERACTIONS}")
print(f"Min book interactions:  {MIN_BOOK_INTERACTIONS}")
print()
print(f"Train — {train_check:>12,} interactions, {train_users_n:>9,} users, {train_books_n:>9,} books")
print(f"Val   — {val_check:>12,} interactions, {val_users_n:>9,} users, {val_books_n:>9,} books")
print(f"Test  — {test_check:>12,} interactions, {test_users_n:>9,} users, {test_books_n:>9,} books")
print()
print(f"Output schema: (user_id: int, book_id: int, is_read: int, rating: int)")
print(f"Output paths:")
print(f"  Train: {OUTPUT_BASE}/train")
print(f"  Val:   {OUTPUT_BASE}/val")
print(f"  Test:  {OUTPUT_BASE}/test")

spark.stop()
