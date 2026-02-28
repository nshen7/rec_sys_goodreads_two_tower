"""
03 — Implicit ALS with Hybrid Interaction Strength (Dataproc version)

Trains implicit ALS with grid search over (rank, regParam, maxIter, beta),
evaluates with ranking metrics (Precision@K, Recall@K, NDCG@K),
and saves best model and results.

Submits via:
    gcloud dataproc batches submit pyspark \
        gs://<BUCKET>/projects/rec_sys_goodreads/notebooks/03_fit_als_full.py \
        --region=us-central1 \
        --properties="spark.driver.memory=16g,spark.executor.memory=16g,spark.executor.cores=4,spark.sql.legacy.timeParserPolicy=LEGACY,spark.driver.maxResultSize=4g,spark.executor.extraJavaOptions=-Xss4m,spark.driver.extraJavaOptions=-Xss4m"
"""

import time
from itertools import product

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS
from pyspark.storagelevel import StorageLevel

# ===================================================================
# Configuration
# ===================================================================
SAMPLE_PCT = 20.0  # Must match the percentage used in 02_data_split_full.py
K = 10

# --- Spark session (Dataproc handles GCS connector, Java, auth) ---
spark = (
    SparkSession.builder
    .appName("goodreads_als")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .getOrCreate()
)

# Custom print function that writes to both stdout and buffer
_original_print = print
def print(*args, **kwargs):
    _original_print(*args, **kwargs)
    _original_print(*args, **kwargs, file=output_buffer)

# Redirect stdout to capture all print statements
from io import StringIO
output_buffer = StringIO()

GCS_BASE = "gs://nshen7-personal-bucket/projects/rec_sys_goodreads"
SPLITS_BASE = f"{GCS_BASE}/data/splits_sample_{int(SAMPLE_PCT)}pct"
MODEL_BASE = f"{GCS_BASE}/models/als_sample_{int(SAMPLE_PCT)}pct"
OUTPUT_LOG = f"{MODEL_BASE}/training_log.txt"

print(f"Configuration: Using {SAMPLE_PCT}% sample data")
print(f"Data path: {SPLITS_BASE}")
print(f"Model output: {MODEL_BASE}")


# ===================================================================
# Helper functions
# ===================================================================
def compute_r(df, beta):
    """Compute interaction strength r = 1 + 2*is_read + beta*max(0, rating - 3)."""
    return df.withColumn(
        "r",
        F.lit(1)
        + F.lit(2) * F.col("is_read")
        + F.lit(beta) * F.greatest(F.lit(0), F.col("rating") - F.lit(3)),
    )


def evaluate_ranking(model, eval_df, beta, k=10):
    """
    Compute Precision@K, Recall@K, NDCG@K (graded) for implicit feedback.

    Ground truth: all (user, book) pairs in eval_df.
    NDCG uses graded relevance: r = 1 + 2*is_read + beta*max(0, rating-3).

    FIX: Added materialization (.count() calls) to break down the logical plan
    and prevent StackOverflowError in Spark's query planner.
    """
    cached = []
    try:
        eval_users = (
            eval_df.select("user_id")
            .distinct()
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        cached.append(eval_users)
        n_eval_users = eval_users.count()

        # Compute graded relevance on eval set (model never saw this data)
        eval_with_r = compute_r(eval_df, beta)
        ground_truth_r = (
            eval_with_r.select("user_id", "book_id", "r")
            .persist(StorageLevel.MEMORY_AND_DISK)  # Changed from DISK_ONLY
        )
        cached.append(ground_truth_r)
        ground_truth_r.count()  # Force materialization

        # Generate top-K recommendations
        recs = model.recommendForUserSubset(eval_users, k)
        recs_exploded = (
            recs
            .select("user_id", F.posexplode("recommendations").alias("rec_rank", "rec"))
            .select("user_id", "rec_rank", F.col("rec.book_id").alias("rec_book_id"))
            .persist(StorageLevel.MEMORY_AND_DISK)  # Changed from DISK_ONLY
        )
        cached.append(recs_exploded)
        recs_exploded.count()  # Force materialization

        # Hits: recommended items that appear in ground truth (with relevance score)
        hits = (
            recs_exploded.join(
                ground_truth_r,
                (recs_exploded["user_id"] == ground_truth_r["user_id"])
                & (recs_exploded["rec_book_id"] == ground_truth_r["book_id"]),
                "inner",
            )
            .select(recs_exploded["user_id"], "rec_book_id", "rec_rank", "r")
            .persist(StorageLevel.MEMORY_AND_DISK)  # Changed from DISK_ONLY
        )
        cached.append(hits)
        hits.count()  # Force materialization

        hits_per_user = (
            hits.groupBy("user_id")
            .agg(F.count("*").alias("n_hits"))
            .persist(StorageLevel.MEMORY_AND_DISK)  # Added caching
        )
        cached.append(hits_per_user)
        hits_per_user.count()  # Force materialization

        n_relevant_per_user = (
            ground_truth_r.groupBy("user_id")
            .agg(F.count("*").alias("n_relevant"))
            .persist(StorageLevel.MEMORY_AND_DISK)  # Added caching
        )
        cached.append(n_relevant_per_user)
        n_relevant_per_user.count()  # Force materialization

        # --- Precision@K ---
        total_hits = hits_per_user.agg(F.sum("n_hits").alias("total_hits")).first()[0] or 0
        avg_precision_at_k = total_hits / (n_eval_users * k)

        # --- Recall@K ---
        recall_per_user = (
            eval_users
            .join(hits_per_user, "user_id", "left")
            .fillna(0, subset=["n_hits"])
            .join(n_relevant_per_user, "user_id", "left")
            .fillna(1, subset=["n_relevant"])
            .withColumn("recall_at_k", F.col("n_hits") / F.col("n_relevant"))
            .persist(StorageLevel.MEMORY_AND_DISK)  # Added caching
        )
        cached.append(recall_per_user)
        recall_per_user.count()  # Force materialization

        sum_recall = recall_per_user.agg(F.sum("recall_at_k").alias("sum_recall")).first()[0]
        avg_recall_at_k = sum_recall / n_eval_users

        # --- NDCG@K (graded relevance) ---
        # DCG: sum r_i / log2(rank + 2) for each hit
        dcg_per_user = (
            hits.select("user_id", "rec_rank", "r")
            .withColumn("dcg_i", F.col("r") / F.log2(F.col("rec_rank") + 2))
            .groupBy("user_id")
            .agg(F.sum("dcg_i").alias("dcg"))
            .persist(StorageLevel.MEMORY_AND_DISK)  # Added caching
        )
        cached.append(dcg_per_user)
        dcg_per_user.count()  # Force materialization

        # IDCG: sort each user's eval items by r descending, take top-K, sum r_i / log2(i+2)
        w_ideal = Window.partitionBy("user_id").orderBy(F.desc("r"))
        ideal_ranking = (
            ground_truth_r
            .withColumn("ideal_rank", F.row_number().over(w_ideal) - 1)  # 0-indexed
            .filter(F.col("ideal_rank") < k)
            .withColumn("idcg_i", F.col("r") / F.log2(F.col("ideal_rank") + 2))
            .groupBy("user_id")
            .agg(F.sum("idcg_i").alias("idcg"))
            .persist(StorageLevel.MEMORY_AND_DISK)  # Added caching
        )
        cached.append(ideal_ranking)
        ideal_ranking.count()  # Force materialization

        ndcg_per_user = (
            eval_users
            .join(dcg_per_user, "user_id", "left")
            .fillna(0.0, subset=["dcg"])
            .join(ideal_ranking, "user_id", "left")
            .fillna(1.0, subset=["idcg"])
            .withColumn("ndcg", F.col("dcg") / F.col("idcg"))
        )
        sum_ndcg = ndcg_per_user.agg(F.sum("ndcg").alias("sum_ndcg")).first()[0]
        avg_ndcg = sum_ndcg / n_eval_users

    finally:
        for df in reversed(cached):
            df.unpersist(blocking=False)

    return {
        f"precision@{k}": avg_precision_at_k,
        f"recall@{k}": avg_recall_at_k,
        f"ndcg@{k}": avg_ndcg,
    }


# ===================================================================
# 1. Load splits
# ===================================================================
train = spark.read.parquet(f"{SPLITS_BASE}/train").cache()
val = spark.read.parquet(f"{SPLITS_BASE}/val").cache()
test = spark.read.parquet(f"{SPLITS_BASE}/test")

n_train = train.count()
n_val = val.count()
n_test = test.count()

print(f"Train: {n_train:,} interactions")
print(f"Val:   {n_val:,} interactions")
print(f"Test:  {n_test:,} interactions")

assert set(train.columns) == {"user_id", "book_id", "is_read", "rating"}, \
    f"Unexpected columns: {train.columns}"

# ===================================================================
# 2. Popularity baseline
# ===================================================================
popular_books = (
    train.groupBy("book_id")
    .agg(F.count("*").alias("n_interactions"))
    .orderBy(F.desc("n_interactions"))
    .limit(K)
    .select("book_id")
    .collect()
)
popular_book_ids = set(row["book_id"] for row in popular_books)

ground_truth_val = val.select("user_id", "book_id")
n_val_users = val.select("user_id").distinct().count()
pop_hits_total = ground_truth_val.filter(F.col("book_id").isin(popular_book_ids)).count()
pop_precision = pop_hits_total / (n_val_users * K)
print(f"\nPopularity baseline Precision@{K}: {pop_precision:.4f}")

# ===================================================================
# 3. Implicit ALS grid search on validation set
# ===================================================================
param_grid = {
    "rank": [50],
    "regParam": [0.1],
    "maxIter": [20],
    "beta": [0.5],
}

results = []
best_model = None
best_ndcg = -1.0

combos = list(product(
    param_grid["rank"],
    param_grid["regParam"],
    param_grid["maxIter"],
    param_grid["beta"],
))

for i, (rank, reg, max_iter, beta) in enumerate(combos, 1):
    print(f"[{i}/{len(combos)}] rank={rank}, regParam={reg}, "
          f"maxIter={max_iter}, beta={beta} ... ", end="", flush=True)
    t0 = time.time()

    train_r = compute_r(train, beta).select(
        "user_id", "book_id", F.col("r").cast("float").alias("rating")
    )

    als = ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=reg,
        userCol="user_id",
        itemCol="book_id",
        ratingCol="rating",
        implicitPrefs=True,
        coldStartStrategy="drop",
        seed=42,
    )

    model = als.fit(train_r)
    metrics = evaluate_ranking(model, val, beta, k=K)

    elapsed = time.time() - t0
    result = {
        "rank": rank,
        "regParam": reg,
        "maxIter": max_iter,
        "beta": beta,
        **metrics,
        "elapsed_s": round(elapsed, 1),
    }
    results.append(result)

    # FIX: Only keep the best model, discard others to save memory
    current_ndcg = metrics[f'ndcg@{K}']
    if current_ndcg > best_ndcg:
        # Unpersist old best model if it exists
        if best_model is not None:
            try:
                best_model.userFactors.unpersist()
                best_model.itemFactors.unpersist()
            except:
                pass
        best_model = model
        best_ndcg = current_ndcg
    else:
        # Unpersist this model since we won't use it
        try:
            model.userFactors.unpersist()
            model.itemFactors.unpersist()
        except:
            pass

    print(f"NDCG@{K}={metrics[f'ndcg@{K}']:.4f}, "
          f"P@{K}={metrics[f'precision@{K}']:.4f} ({elapsed:.0f}s)")

print("\nGrid search complete.")

# ===================================================================
# 4. Best model selection
# ===================================================================
# Sort results by NDCG
results_sorted = sorted(results, key=lambda x: -x[f"ndcg@{K}"])
best = results_sorted[0]

print(f"\n{'Rank':>6} {'RegParam':>10} {'MaxIter':>8} {'Beta':>6} "
      f"{'NDCG@10':>10} {'P@10':>8} {'R@10':>8}")
print("-" * 62)
for r in results_sorted[:20]:
    print(f"{r['rank']:>6} {r['regParam']:>10.3f} {r['maxIter']:>8} {r['beta']:>6.1f} "
          f"{r[f'ndcg@{K}']:>10.4f} {r[f'precision@{K}']:>8.4f} "
          f"{r[f'recall@{K}']:>8.4f}")

print(f"\nBest: rank={best['rank']}, regParam={best['regParam']}, "
      f"maxIter={best['maxIter']}, beta={best['beta']}")
print(f"Best Val NDCG@{K}: {best[f'ndcg@{K}']:.4f}")

# ===================================================================
# 5. Evaluate best model on test set (no retraining)
# ===================================================================
print("\nEvaluating best model on test set...")
best_beta = best["beta"]

# Evaluate at multiple K values
K_VALUES = [10, 50, 100]
test_metrics_all = {}
for k in K_VALUES:
    print(f"  Computing metrics for K={k}...")
    metrics_k = evaluate_ranking(best_model, test, best_beta, k=k)
    test_metrics_all[k] = metrics_k

# ===================================================================
# 6. Save model & results
# ===================================================================
best_model.write().overwrite().save(f"{MODEL_BASE}/model")
print(f"\nModel saved to: {MODEL_BASE}/model")

tuning_df = spark.createDataFrame(results_sorted)
tuning_df.coalesce(1).write.mode("overwrite").parquet(f"{MODEL_BASE}/tuning_results")
print(f"Tuning results saved to: {MODEL_BASE}/tuning_results")

# ===================================================================
# 7. Summary
# ===================================================================
print()
print("=" * 60)
print("  IMPLICIT ALS MODEL SUMMARY")
print("=" * 60)
print(f"Configuration:")
print(f"  Sample percentage: {SAMPLE_PCT}%")
print()
print(f"Data:")
print(f"  Train:             {n_train:,} interactions")
print(f"  Val:               {n_val:,} interactions")
print(f"  Test:              {n_test:,} interactions")
print()
print(f"Best hyperparameters:")
print(f"  rank:              {best['rank']}")
print(f"  regParam:          {best['regParam']}")
print(f"  maxIter:           {best['maxIter']}")
print(f"  beta:              {best['beta']}")
print()
print(f"Interaction strength: r = 1 + 2*is_read + {best['beta']}*max(0, rating-3)")
print()
print(f"Validation metrics (K={K}):")
print(f"  NDCG@{K}:           {best[f'ndcg@{K}']:.4f}")
print(f"  Precision@{K}:      {best[f'precision@{K}']:.4f}")
print(f"  Recall@{K}:         {best[f'recall@{K}']:.4f}")
print()
print(f"Test metrics:")
for k in K_VALUES:
    print(f"  K={k}:")
    for metric_name, metric_val in test_metrics_all[k].items():
        print(f"    {metric_name}:{'':>{14-len(metric_name)}}{metric_val:.4f}")
print()
print(f"Output paths:")
print(f"  Model:             {MODEL_BASE}/model")
print(f"  Tuning results:    {MODEL_BASE}/tuning_results")
print(f"  Training log:      {OUTPUT_LOG}")

# Save all output to text file
output_text = output_buffer.getvalue()
output_rdd = spark.sparkContext.parallelize([output_text])
output_rdd.saveAsTextFile(OUTPUT_LOG)
print(f"\nTraining log saved to: {OUTPUT_LOG}")

spark.stop()
