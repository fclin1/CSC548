#!/bin/bash
# ==============================================================================
# setup_arc.sh - Environment Verification & Smoke Test for HW4 Spark SpMV
# Target Hardware / Software: ARC Compute Node (Spark 3.5.0, Python 3.11, Java 8)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo " HW4 Spark SpMV Environment Verification on $(hostname)"
echo " Date: $(date)"
echo " Directory: $SCRIPT_DIR"
echo "=================================================================="

# 1. Environment Exports
export SPARK_HOME="${SPARK_HOME:-/usr/local/spark}"
export PATH="$SPARK_HOME/bin:$PATH"
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-/usr/bin/python3.11}"
export PYSPARK_DRIVER_PYTHON="${PYSPARK_DRIVER_PYTHON:-$PYSPARK_PYTHON}"

echo ""
echo "[1/4] Checking Python..."
if command -v "$PYSPARK_PYTHON" &> /dev/null; then
    echo "Found Python: $("$PYSPARK_PYTHON" --version)"
else
    echo "[WARNING] $PYSPARK_PYTHON not found, falling back to $(which python3)"
    export PYSPARK_PYTHON="$(which python3)"
    export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
fi

echo ""
echo "[2/4] Checking Java..."
if command -v java &> /dev/null; then
    java -version 2>&1 | head -n 2
else
    echo "[ERROR] Java not found in PATH!"
    exit 1
fi

echo ""
echo "[3/4] Checking Matrices..."
REQUIRED_MATRICES=(
    "D6-6.mtx"
    "dictionary28.mtx"
    "bfly.mtx"
    "Ga3As3H12.mtx"
    "pkustk14.mtx"
    "roadNet-CA.mtx"
)

matrices_ready=true
if [ ! -d "matrices" ]; then
    matrices_ready=false
else
    for m in "${REQUIRED_MATRICES[@]}"; do
        if [ ! -f "matrices/$m" ]; then
            matrices_ready=false
            break
        fi
    done
fi

if [ "$matrices_ready" = false ]; then
    echo "Matrix suite not ready in 'matrices/'. Searching candidate locations..."
    found_dir=""
    for cand in \
        "$SCRIPT_DIR/../../hw3/SpMV/matrices" \
        "$SCRIPT_DIR/../../hw2/SpMV/matrices" \
        "$SCRIPT_DIR/../../hw1/SpMV/matrices" \
        "$HOME/CSC548/hw3/SpMV/matrices" \
        "$HOME/CSC548/hw2/SpMV/matrices" \
        "$HOME/CSC548/hw1/SpMV/matrices" \
        "/share/$USER/matrices"; do
        if [ -d "$cand" ] && [ -f "$cand/D6-6.mtx" ] && [ -f "$cand/pkustk14.mtx" ]; then
            found_dir="$(cd "$cand" && pwd -P)"
            break
        fi
    done

    if [ -n "$found_dir" ]; then
        echo "Found matrix dataset at: $found_dir"
        rm -rf matrices 2>/dev/null || true
        ln -sfn "$found_dir" matrices
        echo "Created symlink: matrices -> $found_dir"
    else
        echo ""
        echo "[ERROR] Benchmark matrices not found!"
        echo "To upload matrices from your local machine to ARC, run from your local terminal:"
        echo "  bash hw4/sync_arc.sh push fclin --with-matrices"
        exit 1
    fi
fi

for m in "${REQUIRED_MATRICES[@]}"; do
    if [ -f "matrices/$m" ]; then
        echo "  [OK] matrices/$m ($(ls -lh "matrices/$m" | awk '{print $5}'))"
    else
        echo "  [MISSING] matrices/$m"
    fi
done

echo ""
echo "[4/4] Running Quick Smoke Test on D6-6.mtx..."
for scheme in join broadcast; do
    echo "Testing scheme: $scheme..."
    spark-submit --master "local[2]" --driver-memory "4g" \
        spmv-spark.py "matrices/D6-6.mtx" --scheme "$scheme" --partitions 2 --iters 1 --warmup 0
done

echo ""
echo "=================================================================="
echo " [SUCCESS] Environment and kernels verified! Ready to launch sweep:"
echo "   sbatch sweep.sbatch"
echo "=================================================================="
