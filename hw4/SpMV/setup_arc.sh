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
mkdir -p matrices
for m in D6-6 dictionary28 bfly roadNet-CA Ga3As3H12 pkustk14; do
    if [ ! -f "matrices/$m.mtx" ]; then
        for prev in ../../hw3/SpMV/matrices ../../hw2/SpMV/matrices ../../hw1/SpMV/matrices; do
            if [ -f "$prev/$m.mtx" ]; then
                echo "Linking $m.mtx from $prev..."
                ln -sf "$prev/$m.mtx" "matrices/$m.mtx"
                break
            fi
        done
    fi
    if [ -f "matrices/$m.mtx" ]; then
        echo "  [OK] matrices/$m.mtx ($(ls -lh "matrices/$m.mtx" | awk '{print $5}'))"
    else
        echo "  [MISSING] matrices/$m.mtx"
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
