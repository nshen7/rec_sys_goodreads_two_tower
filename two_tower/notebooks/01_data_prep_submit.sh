#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Submit 01_data_prep.py to Dataproc
#
# All job parameters use their Python defaults unless overridden here.
# Override GCP config via environment variables if needed:
#   GCP_PROJECT, DATAPROC_REGION, DATAPROC_CLUSTER
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ID="project-42495fb5-90f0-4a7f-bf3"          # <-- REPLACE with your GCP project ID
REGION="us-central1"
BATCH_ID="goodreads-twotower-$(date +%Y%m%d-%H%M%S)"

GCS_BASE="gs://nshen7-personal-bucket/projects/rec_sys_goodreads"

SCRIPT_LOCAL="$(dirname "$0")/01_data_prep.py"
SCRIPT_GCS="${GCS_BASE}/two_tower/jobs/01_data_prep.py"

# ---------------------------------------------------------------------------
# Upload script to GCS
# ---------------------------------------------------------------------------
echo "Uploading script to GCS..."
gsutil cp "$SCRIPT_LOCAL" "$SCRIPT_GCS"
echo "  -> $SCRIPT_GCS"

# ---------------------------------------------------------------------------
# Submit Dataproc job
# ---------------------------------------------------------------------------
echo ""
echo "Submitting Dataproc PySpark job..."
echo "  Project : $PROJECT_ID"
echo "  Region  : $REGION"
echo "  Batch   : $BATCH_ID"
echo ""

gcloud dataproc batches submit pyspark "$SCRIPT_GCS" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --batch="${BATCH_ID}" \
    --async \
    --properties="\
spark.driver.memory=16g,\
spark.driver.memoryOverhead=4g,\
spark.executor.memory=16g,\
spark.executor.memoryOverhead=6g,\
spark.executor.cores=4,\
spark.driver.maxResultSize=4g,\
spark.executor.extraJavaOptions=-Xss4m,\
spark.driver.extraJavaOptions=-Xss4m,\
spark.sql.shuffle.partitions=800,\
spark.default.parallelism=800,\
spark.sql.adaptive.enabled=true,\
spark.sql.adaptive.skewJoin.enabled=true,\
spark.sql.adaptive.coalescePartitions.enabled=true,\
spark.sql.legacy.timeParserPolicy=LEGACY"
