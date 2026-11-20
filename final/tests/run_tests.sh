#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

SUMMARY_FILE="results/test_suite_summary.txt"
TINY_LOG="results/tiny_check.txt"
mkdir -p results data

echo "=== CSC 548 AUTOMATED K-MEANS VERIFICATION SUITE ===" | tee "${SUMMARY_FILE}"
echo "Started at $(date -u)" | tee -a "${SUMMARY_FILE}"

# Generate test fixtures
gcc -O2 tests/generate_fixtures.c -o bin/generate_fixtures -lm
./bin/generate_fixtures

PASSED_GROUPS=0
TOTAL_GROUPS=8

run_check() {
    local cmd="$1"
    local desc="$2"
    echo -n "  Testing: ${desc}... "
    if eval "${cmd}" > /dev/null 2>&1; then
        echo "PASS"
        return 0
    else
        echo "FAIL (command: ${cmd})"
        return 1
    fi
}

# -----------------------------------------------------------------------------
echo "" | tee -a "${SUMMARY_FILE}"
echo "[Test Group 1/8] T1 — Tiny Exact Ground Truth (N=1024, D=8, K=4, I=20)" | tee -a "${SUMMARY_FILE}"
T1_FAIL=0
./bin/kmeans_seq --synthetic --n 1024 --d 8 --k 4 --iters 20 --dump-ref data/ref_t1 > /dev/null
./bin/kmeans_seq --synthetic --n 1024 --d 8 --k 4 --iters 20 --check-ref data/ref_t1 | tee "${TINY_LOG}" > /dev/null || T1_FAIL=1
./bin/kmeans_cuda_naive --synthetic --n 1024 --d 8 --k 4 --iters 20 --check-ref data/ref_t1 >> "${TINY_LOG}" 2>&1 || T1_FAIL=1
./bin/kmeans_cuda_shmem --synthetic --n 1024 --d 8 --k 4 --iters 20 --check-ref data/ref_t1 >> "${TINY_LOG}" 2>&1 || T1_FAIL=1
./bin/kmeans_cuda_device --synthetic --n 1024 --d 8 --k 4 --iters 20 --check-ref data/ref_t1 >> "${TINY_LOG}" 2>&1 || T1_FAIL=1

if [ ${T1_FAIL} -eq 0 ]; then
    echo "  Group 1: PASS (all 4 binaries matched reference bit-accurately)" | tee -a "${SUMMARY_FILE}"
    PASSED_GROUPS=$((PASSED_GROUPS + 1))
else
    echo "  Group 1: FAIL" | tee -a "${SUMMARY_FILE}"
fi

# -----------------------------------------------------------------------------
echo "" | tee -a "${SUMMARY_FILE}"
echo "[Test Group 2/8] T2 — Medium Realistic Parity (N=65536, D=18, K=32, I=20)" | tee -a "${SUMMARY_FILE}"
T2_FAIL=0
./bin/kmeans_seq --synthetic --n 65536 --d 18 --k 32 --iters 20 --dump-ref data/ref_t2 > /dev/null
./bin/kmeans_cuda_naive --synthetic --n 65536 --d 18 --k 32 --iters 20 --check-ref data/ref_t2 > /dev/null 2>&1 || T2_FAIL=1
./bin/kmeans_cuda_shmem --synthetic --n 65536 --d 18 --k 32 --iters 20 --check-ref data/ref_t2 > /dev/null 2>&1 || T2_FAIL=1
./bin/kmeans_cuda_device --synthetic --n 65536 --d 18 --k 32 --iters 20 --check-ref data/ref_t2 > /dev/null 2>&1 || T2_FAIL=1

if [ ${T2_FAIL} -eq 0 ]; then
    echo "  Group 2: PASS (all 3 CUDA stages match serial reference within tolerance)" | tee -a "${SUMMARY_FILE}"
    PASSED_GROUPS=$((PASSED_GROUPS + 1))
else
    echo "  Group 2: FAIL" | tee -a "${SUMMARY_FILE}"
fi

# -----------------------------------------------------------------------------
echo "" | tee -a "${SUMMARY_FILE}"
echo "[Test Group 3/8] T3 — Shared-Memory Tiling Boundary (N=16384, D=18, K=1024, I=5)" | tee -a "${SUMMARY_FILE}"
T3_FAIL=0
./bin/kmeans_seq --synthetic --n 16384 --d 18 --k 1024 --iters 5 --dump-ref data/ref_t3 > /dev/null
./bin/kmeans_cuda_naive --synthetic --n 16384 --d 18 --k 1024 --iters 5 --check-ref data/ref_t3 > /dev/null 2>&1 || T3_FAIL=1
./bin/kmeans_cuda_shmem --synthetic --n 16384 --d 18 --k 1024 --iters 5 --check-ref data/ref_t3 > /dev/null 2>&1 || T3_FAIL=1
./bin/kmeans_cuda_device --synthetic --n 16384 --d 18 --k 1024 --iters 5 --check-ref data/ref_t3 > /dev/null 2>&1 || T3_FAIL=1

if [ ${T3_FAIL} -eq 0 ]; then
    echo "  Group 3: PASS (2-pass 48 KB centroid tiling and Path B reduction verified)" | tee -a "${SUMMARY_FILE}"
    PASSED_GROUPS=$((PASSED_GROUPS + 1))
else
    echo "  Group 3: FAIL" | tee -a "${SUMMARY_FILE}"
fi

# -----------------------------------------------------------------------------
echo "" | tee -a "${SUMMARY_FILE}"
echo "[Test Group 4/8] T4 — Prime Non-Multiple-of-128 & Tail Masking (N=10007, D=13, K=7, I=10)" | tee -a "${SUMMARY_FILE}"
T4_FAIL=0
./bin/kmeans_seq --synthetic --n 10007 --d 13 --k 7 --iters 10 --dump-ref data/ref_t4 > /dev/null
./bin/kmeans_cuda_naive --synthetic --n 10007 --d 13 --k 7 --iters 10 --check-ref data/ref_t4 > /dev/null 2>&1 || T4_FAIL=1
./bin/kmeans_cuda_shmem --synthetic --n 10007 --d 13 --k 7 --iters 10 --check-ref data/ref_t4 > /dev/null 2>&1 || T4_FAIL=1
./bin/kmeans_cuda_device --synthetic --n 10007 --d 13 --k 7 --iters 10 --check-ref data/ref_t4 > /dev/null 2>&1 || T4_FAIL=1

if [ ${T4_FAIL} -eq 0 ]; then
    echo "  Group 4: PASS (float4 tail masking and SoA padding verified on prime sizes)" | tee -a "${SUMMARY_FILE}"
    PASSED_GROUPS=$((PASSED_GROUPS + 1))
else
    echo "  Group 4: FAIL" | tee -a "${SUMMARY_FILE}"
fi

# -----------------------------------------------------------------------------
echo "" | tee -a "${SUMMARY_FILE}"
echo "[Test Group 5/8] T5 — Forced Empty-Cluster Edge Case (data/test_empty_cluster.bin, K=4)" | tee -a "${SUMMARY_FILE}"
T5_FAIL=0
./bin/kmeans_seq --file data/test_empty_cluster.bin --k 4 --iters 5 --dump-ref data/ref_t5 > /dev/null
./bin/kmeans_cuda_naive --file data/test_empty_cluster.bin --k 4 --iters 5 --check-ref data/ref_t5 > /dev/null 2>&1 || T5_FAIL=1
./bin/kmeans_cuda_shmem --file data/test_empty_cluster.bin --k 4 --iters 5 --check-ref data/ref_t5 > /dev/null 2>&1 || T5_FAIL=1
./bin/kmeans_cuda_device --file data/test_empty_cluster.bin --k 4 --iters 5 --check-ref data/ref_t5 > /dev/null 2>&1 || T5_FAIL=1

if [ ${T5_FAIL} -eq 0 ]; then
    echo "  Group 5: PASS (zero NaNs and deterministic empty-cluster fallback verified)" | tee -a "${SUMMARY_FILE}"
    PASSED_GROUPS=$((PASSED_GROUPS + 1))
else
    echo "  Group 5: FAIL" | tee -a "${SUMMARY_FILE}"
fi

# -----------------------------------------------------------------------------
echo "" | tee -a "${SUMMARY_FILE}"
echo "[Test Group 6/8] T6 — Exact Tie-Breaking Determinism (data/test_ties.bin, K=2)" | tee -a "${SUMMARY_FILE}"
T6_FAIL=0
./bin/kmeans_seq --file data/test_ties.bin --k 2 --iters 5 --dump-ref data/ref_t6 > /dev/null
./bin/kmeans_cuda_naive --file data/test_ties.bin --k 2 --iters 5 --check-ref data/ref_t6 > /dev/null 2>&1 || T6_FAIL=1
./bin/kmeans_cuda_shmem --file data/test_ties.bin --k 2 --iters 5 --check-ref data/ref_t6 > /dev/null 2>&1 || T6_FAIL=1
./bin/kmeans_cuda_device --file data/test_ties.bin --k 2 --iters 5 --check-ref data/ref_t6 > /dev/null 2>&1 || T6_FAIL=1

if [ ${T6_FAIL} -eq 0 ]; then
    echo "  Group 6: PASS (deterministic tie breaking chooses index 0 across all stages)" | tee -a "${SUMMARY_FILE}"
    PASSED_GROUPS=$((PASSED_GROUPS + 1))
else
    echo "  Group 6: FAIL" | tee -a "${SUMMARY_FILE}"
fi

# -----------------------------------------------------------------------------
echo "" | tee -a "${SUMMARY_FILE}"
echo "[Test Group 7/8] T7 — Early-Stopping & Warp-Shuffle Convergence Test" | tee -a "${SUMMARY_FILE}"
T7_FAIL=0
./bin/kmeans_seq --synthetic --n 8192 --d 8 --k 8 --iters 100 --early-stop --tol 1e-4 --dump-ref data/ref_t7 > /dev/null
./bin/kmeans_cuda_naive --synthetic --n 8192 --d 8 --k 8 --iters 100 --early-stop --tol 1e-4 --check-ref data/ref_t7 > /dev/null 2>&1 || T7_FAIL=1
./bin/kmeans_cuda_shmem --synthetic --n 8192 --d 8 --k 8 --iters 100 --early-stop --tol 1e-4 --check-ref data/ref_t7 > /dev/null 2>&1 || T7_FAIL=1
./bin/kmeans_cuda_device --synthetic --n 8192 --d 8 --k 8 --iters 100 --early-stop --tol 1e-4 --check-ref data/ref_t7 > /dev/null 2>&1 || T7_FAIL=1

if [ ${T7_FAIL} -eq 0 ]; then
    echo "  Group 7: PASS (warp-shuffle convergence reduction and early stopping matched)" | tee -a "${SUMMARY_FILE}"
    PASSED_GROUPS=$((PASSED_GROUPS + 1))
else
    echo "  Group 7: FAIL" | tee -a "${SUMMARY_FILE}"
fi

# -----------------------------------------------------------------------------
echo "" | tee -a "${SUMMARY_FILE}"
echo "[Test Group 8/8] T8 — Invalid Input Rejection (Negative Assertions)" | tee -a "${SUMMARY_FILE}"
T8_FAIL=0

# Subtest 1: N < K (exit code 3)
set +e
./bin/kmeans_seq --synthetic --n 10 --k 20 > /dev/null 2>&1
RC1=$?
set -e
if [ ${RC1} -ne 3 ]; then T8_FAIL=1; fi

# Subtest 2: D = 0 (exit code 1 or 3)
set +e
./bin/kmeans_seq --synthetic --n 100 --d 0 > /dev/null 2>&1
RC2=$?
set -e
if [ ${RC2} -ne 1 ] && [ ${RC2} -ne 3 ]; then T8_FAIL=1; fi

# Subtest 3: block_size not multiple of 32 (exit code 1)
set +e
./bin/kmeans_cuda_naive --synthetic --n 1000 --k 10 --block-size 100 > /dev/null 2>&1
RC3=$?
set -e
if [ ${RC3} -ne 1 ]; then T8_FAIL=1; fi

# Subtest 4: NaN in input data (exit code 3)
set +e
./bin/kmeans_seq --file data/test_nan.bin --k 2 > /dev/null 2>&1
RC4=$?
set -e
if [ ${RC4} -ne 3 ]; then T8_FAIL=1; fi

# Subtest 5: Missing input file (exit code 2)
set +e
./bin/kmeans_seq --file data/nonexistent_dataset.bin --k 2 > /dev/null 2>&1
RC5=$?
set -e
if [ ${RC5} -ne 2 ]; then T8_FAIL=1; fi

if [ ${T8_FAIL} -eq 0 ]; then
    echo "  Group 8: PASS (all 5 negative validation tests rejected with expected exit codes)" | tee -a "${SUMMARY_FILE}"
    PASSED_GROUPS=$((PASSED_GROUPS + 1))
else
    echo "  Group 8: FAIL (RCs: N<K=${RC1}, D=0=${RC2}, block=${RC3}, NaN=${RC4}, missing=${RC5})" | tee -a "${SUMMARY_FILE}"
fi

# -----------------------------------------------------------------------------
echo "" | tee -a "${SUMMARY_FILE}"
echo "=============================================================================" | tee -a "${SUMMARY_FILE}"
echo "TEST SUITE RESULT: ${PASSED_GROUPS}/${TOTAL_GROUPS} GROUPS PASSED" | tee -a "${SUMMARY_FILE}"
echo "=============================================================================" | tee -a "${SUMMARY_FILE}"

# Clean up temp test references
rm -f data/ref_t* bin/generate_fixtures

if [ ${PASSED_GROUPS} -eq ${TOTAL_GROUPS} ]; then
    exit 0
else
    exit 1
fi
