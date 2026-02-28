#!/bin/bash

# Download core Goodreads datasets and upload directly to GCS
# Usage: ./download_to_gcs.sh

set -e

BUCKET="nshen7-personal-bucket"
GCS_RAW="gs://${BUCKET}/projects/rec_sys_goodreads/data/raw"
BASE_URL="https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"

# Core datasets used by the project
FILES=(
    "goodreads_books.json.gz"
    "goodreads_interactions.csv"
    "goodreads_interactions_dedup.json.gz"
    "goodreads_reviews_dedup.json.gz"
    "book_id_map.csv"
    "user_id_map.csv"
    "book_clubs.json"
)

TMPDIR=$(mktemp -d)
trap 'rm -rf "${TMPDIR}"' EXIT

echo "=== Downloading datasets to GCS ==="
echo "Destination: ${GCS_RAW}"
echo "Temp dir:    ${TMPDIR}"
echo ""

for FILE in "${FILES[@]}"; do
    echo "--- ${FILE} ---"
    LOCAL_PATH="${TMPDIR}/${FILE}"

    echo "  Downloading from ${BASE_URL}/${FILE} ..."
    curl -L -o "${LOCAL_PATH}" "${BASE_URL}/${FILE}"

    echo "  Uploading to ${GCS_RAW}/${FILE} ..."
    gsutil -m cp "${LOCAL_PATH}" "${GCS_RAW}/${FILE}"

    # Remove local copy to save disk space (some files are very large)
    rm -f "${LOCAL_PATH}"

    echo "  Done."
    echo ""
done

echo "=== All downloads complete ==="
echo "Files in GCS:"
gsutil ls -l "${GCS_RAW}/"
