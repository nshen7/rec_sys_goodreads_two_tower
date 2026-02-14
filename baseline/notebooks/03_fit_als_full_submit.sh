#!/usr/bin/env bash
#
# Submit the ALS training PySpark job to Dataproc Serverless.
#
# Usage:
#   bash notebooks/03_fit_als_full_submit.sh
#
set -euo pipefail

# ---- Configuration (edit these) ----
PROJECT_ID="project-42495fb5-90f0-4a7f-bf3"          # <-- REPLACE with your GCP project ID
REGION="us-central1"
BUCKET="nshen7-personal-bucket"
SCRIPT_LOCAL="/home/s38976581_gmail_com/projects/rec_sys_goodreads/notebooks/03_fit_als_full.py"
SCRIPT_GCS="gs://${BUCKET}/projects/rec_sys_goodreads/notebooks/03_fit_als_full.py"
BATCH_ID="goodreads-als-$(date +%Y%m%d-%H%M%S)"

# ---- Pre-flight checks ----

echo "=== Configuration ==="
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Batch ID: ${BATCH_ID}"
echo ""

# Set project
gcloud config set project "${PROJECT_ID}" --quiet

# Enable Dataproc API (no-op if already enabled)
echo "Enabling Dataproc API..."
gcloud services enable dataproc.googleapis.com --quiet

# ---- Upload script to GCS ----
echo "Uploading script to GCS..."
gsutil cp "${SCRIPT_LOCAL}" "${SCRIPT_GCS}"
echo "  Uploaded: ${SCRIPT_GCS}"
echo ""

# ---- Submit the Serverless batch job ----
echo "Submitting Dataproc Serverless batch job..."
gcloud dataproc batches submit pyspark "${SCRIPT_GCS}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --batch="${BATCH_ID}" \
    --async \
    --properties="\
spark.driver.memory=8g,\
spark.driver.memoryOverhead=4g,\
spark.executor.memory=12g,\
spark.executor.memoryOverhead=6g,\
spark.executor.cores=4,\
spark.sql.shuffle.partitions=800,\
spark.default.parallelism=800,\
spark.sql.adaptive.enabled=true,\
spark.sql.adaptive.skewJoin.enabled=true,\
spark.sql.adaptive.coalescePartitions.enabled=true,\
spark.sql.legacy.timeParserPolicy=LEGACY"

echo ""
echo "=== Job submitted ==="
echo "  Batch ID: ${BATCH_ID}"
echo ""
echo "Monitor with:"
echo "  gcloud dataproc batches describe ${BATCH_ID} --region=${REGION} --project=${PROJECT_ID}"
echo ""
echo "Or view in the console:"
echo "  https://console.cloud.google.com/dataproc/batches?project=${PROJECT_ID}"
