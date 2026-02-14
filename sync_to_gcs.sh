#!/bin/bash

# Sync project directory to Google Cloud Storage
# Usage: ./sync_to_gcs.sh [bucket_name]

BUCKET_NAME="nshen7-personal-bucket"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
GCS_PATH="gs://${BUCKET_NAME}/projects/rec_sys_goodreads"

# Patterns to exclude
EXCLUDE_PATTERN='\.git/|__pycache__/|\.pyc$|\.env$|\.venv/|node_modules/|\.DS_Store|sync_to_gcs\.sh$'

echo "=== GCS Sync ==="
echo "Source:  ${PROJECT_DIR}"
echo "Dest:    ${GCS_PATH}"
echo "Exclude: ${EXCLUDE_PATTERN}"
echo ""

# Dry run first
echo "--- Dry Run ---"
gsutil -m rsync -r -n -x "${EXCLUDE_PATTERN}" "${PROJECT_DIR}" "${GCS_PATH}"

echo ""
read -p "Proceed with sync? (y/n): " CONFIRM
if [[ "${CONFIRM}" != "y" ]]; then
    echo "Sync cancelled."
    exit 0
fi

# Actual sync
echo "--- Syncing ---"
gsutil -m rsync -r -x "${EXCLUDE_PATTERN}" "${PROJECT_DIR}" "${GCS_PATH}"

echo ""
echo "Sync complete."
