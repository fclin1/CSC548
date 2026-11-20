#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
BIN_FILE="${DATA_DIR}/susy.bin"
EXPECTED_BYTES=360000016
CSV_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz"

mkdir -p "${DATA_DIR}"

if [ -f "${BIN_FILE}" ]; then
    ACTUAL_BYTES=$(stat -c%s "${BIN_FILE}" 2>/dev/null || stat -f%z "${BIN_FILE}")
    if [ "${ACTUAL_BYTES}" -eq "${EXPECTED_BYTES}" ]; then
        echo "SUSY binary dataset already exists at ${BIN_FILE} (${ACTUAL_BYTES} bytes). Verification passed."
        exit 0
    else
        echo "Existing ${BIN_FILE} size (${ACTUAL_BYTES}) != expected (${EXPECTED_BYTES}). Rebuilding..."
        rm -f "${BIN_FILE}"
    fi
fi

if [ ! -f "${ROOT_DIR}/bin/csv_to_bin" ]; then
    echo "Building bin/csv_to_bin..."
    make -C "${ROOT_DIR}" bin/csv_to_bin
fi

echo "Downloading and streaming UCI SUSY dataset into binary format..."
if command -v curl >/dev/null 2>&1; then
    curl -sSL "${CSV_URL}" | gzip -dc | "${ROOT_DIR}/bin/csv_to_bin" --in - --out "${BIN_FILE}" --skip-cols 1
elif command -v wget >/dev/null 2>&1; then
    wget -qO- "${CSV_URL}" | gzip -dc | "${ROOT_DIR}/bin/csv_to_bin" --in - --out "${BIN_FILE}" --skip-cols 1
else
    echo "Error: Neither curl nor wget is available." >&2
    exit 1
fi

ACTUAL_BYTES=$(stat -c%s "${BIN_FILE}" 2>/dev/null || stat -f%z "${BIN_FILE}")
if [ "${ACTUAL_BYTES}" -eq "${EXPECTED_BYTES}" ]; then
    echo "Successfully generated and verified ${BIN_FILE} (${ACTUAL_BYTES} bytes)."
else
    echo "Error: ${BIN_FILE} size mismatch: got ${ACTUAL_BYTES}, expected ${EXPECTED_BYTES}." >&2
    exit 2
fi
