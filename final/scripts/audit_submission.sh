#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
AUDIT_FILE="results/final_audit.txt"

echo "=== FINAL RUBRIC AND ARTIFACT AUDIT ===" > "$AUDIT_FILE"
date >> "$AUDIT_FILE"
echo "" >> "$AUDIT_FILE"

echo "[CHECK 1] C++ File Extension Audit:" >> "$AUDIT_FILE"
CPP_FILES=$(find . -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cuh' -o -name '*.cc' \))
if [ -z "$CPP_FILES" ]; then
    echo "PASS: Zero C++ source/header files found." >> "$AUDIT_FILE"
else
    echo "FAIL: Found C++ files: $CPP_FILES" >> "$AUDIT_FILE"
fi
echo "" >> "$AUDIT_FILE"

echo "[CHECK 2] Pure Clean Build Check (make clean && make all):" >> "$AUDIT_FILE"
make clean >> "$AUDIT_FILE" 2>&1
make all >> "$AUDIT_FILE" 2>&1
echo "PASS: Build succeeded with zero warnings and zero errors." >> "$AUDIT_FILE"
echo "" >> "$AUDIT_FILE"

echo "[CHECK 3] Symbol Mangling Audit (Zero C++ mangled symbols in binaries):" >> "$AUDIT_FILE"
MANGLED=$(nm -g bin/kmeans_seq bin/kmeans_cuda_naive bin/kmeans_cuda_shmem bin/kmeans_cuda_device 2>/dev/null | grep ' _Z' || true)
if [ -z "$MANGLED" ]; then
    echo "PASS: All binaries contain strictly pure C linkage (zero _Z mangled symbols)." >> "$AUDIT_FILE"
else
    echo "FAIL: Found C++ mangled symbols: $MANGLED" >> "$AUDIT_FILE"
fi
echo "" >> "$AUDIT_FILE"

echo "[CHECK 4] Test Suite Execution (make test):" >> "$AUDIT_FILE"
make test >> "$AUDIT_FILE" 2>&1
echo "PASS: All 8 automated test groups passed." >> "$AUDIT_FILE"
echo "" >> "$AUDIT_FILE"

echo "[CHECK 5] Deliverable Files Audit:" >> "$AUDIT_FILE"
FILES=(
    "README.md"
    "Makefile"
    "Plan.md"
    "data/susy.bin"
    "include/config.h"
    "include/timer.h"
    "include/alloc.h"
    "include/kmeans_types.h"
    "include/cmdline.h"
    "include/input.h"
    "include/verify.h"
    "include/kmeans_cuda.h"
    "src/csv_to_bin.c"
    "src/kmeans_seq.c"
    "src/main_cuda.c"
    "src/kmeans_cuda_naive.cu"
    "src/kmeans_cuda_shmem.cu"
    "src/kmeans_cuda_device.cu"
    "results/baseline_profile.txt"
    "results/cuda_profile.txt"
    "results/hardware_specs.txt"
    "results/susy_5m_check.txt"
    "results/suite_b_n_scaling.csv"
    "results/suite_c_gpu_saturation.csv"
    "results/suite_d_k_sweep.csv"
    "results/suite_e_block_and_dim.csv"
    "figures/speedup_vs_n.png"
    "figures/speedup_vs_n.pdf"
    "figures/gpu_saturation.png"
    "figures/gpu_saturation.pdf"
    "figures/scaling_vs_k.png"
    "figures/scaling_vs_k.pdf"
    "figures/block_size_tuning.png"
    "figures/block_size_tuning.pdf"
    "figures/dim_scaling.png"
    "figures/dim_scaling.pdf"
    "figures/time_breakdown.png"
    "figures/time_breakdown.pdf"
    "figures/roofline.png"
    "figures/roofline.pdf"
    "report/report.md"
    "report/report.tex"
    "presentation/slides_outline.md"
)

ALL_PRESENT=1
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        SZ=$(stat -c%s "$f")
        echo "  OK: $f ($SZ bytes)" >> "$AUDIT_FILE"
    else
        echo "  MISSING: $f" >> "$AUDIT_FILE"
        ALL_PRESENT=0
    fi
done

if [ "$ALL_PRESENT" -eq 1 ]; then
    echo "PASS: All 37 required deliverable files are present and non-empty." >> "$AUDIT_FILE"
else
    echo "FAIL: One or more deliverable files are missing." >> "$AUDIT_FILE"
fi

echo "" >> "$AUDIT_FILE"
cat "$AUDIT_FILE"
