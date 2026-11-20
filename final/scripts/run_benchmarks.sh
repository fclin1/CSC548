#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p results data

SUSY_BIN="data/susy.bin"
SUITE_B="results/suite_b_n_scaling.csv"
SUITE_C="results/suite_c_gpu_saturation.csv"
SUITE_D="results/suite_d_k_sweep.csv"
SUITE_E="results/suite_e_block_and_dim.csv"
CHECK_5M="results/susy_5m_check.txt"

# Reset CSV files with fresh headers
./bin/kmeans_seq --csv-header > "${SUITE_B}"
./bin/kmeans_seq --csv-header > "${SUITE_C}"
./bin/kmeans_seq --csv-header > "${SUITE_D}"
./bin/kmeans_seq --csv-header > "${SUITE_E}"

echo "================================================================================"
echo "STARTING FULL BENCHMARK MATRIX EXECUTION"
echo "Target: Intel i5-9600K CPU + NVIDIA GTX 1080 Ti GPU"
echo "Dataset: UCI SUSY (${SUSY_BIN})"
echo "================================================================================"

# -----------------------------------------------------------------------------
echo ""
echo "=== SUITE B: Problem Size N Scaling (100k to 5M, D=18, K=32, I=20) ==="
# N in {100000, 500000, 1000000, 2500000, 5000000}
for n in 100000 500000 1000000 2500000 5000000; do
    echo "--- Suite B: N=${n} ---"
    # For kmeans_seq, if N >= 2500000, use 0 warmup and 3 trials to bound serial runtime
    if [ ${n} -ge 2500000 ]; then
        ./bin/kmeans_seq --file "${SUSY_BIN}" --n ${n} --k 32 --iters 20 --warmup 0 --trials 3 --csv "${SUITE_B}"
    else
        ./bin/kmeans_seq --file "${SUSY_BIN}" --n ${n} --k 32 --iters 20 --warmup 1 --trials 5 --csv "${SUITE_B}"
    fi

    for cuda_impl in kmeans_cuda_naive kmeans_cuda_shmem kmeans_cuda_device; do
        ./bin/${cuda_impl} --file "${SUSY_BIN}" --n ${n} --k 32 --iters 20 --block-size 256 --warmup 1 --trials 5 --csv "${SUITE_B}"
    done
done

# -----------------------------------------------------------------------------
echo ""
echo "=== SUITE C1: SM Wave Grid Saturation (kmeans_cuda_device, D=18, K=32) ==="
# N in {7168, 28672, 114688, 458752, 1835008, 5000000}
for n in 7168 28672 114688 458752 1835008 5000000; do
    echo "--- Suite C1 (Wave Saturation): N=${n} ---"
    ./bin/kmeans_cuda_device --file "${SUSY_BIN}" --n ${n} --k 32 --iters 20 --block-size 256 --warmup 1 --trials 5 --csv "${SUITE_C}"
done

echo ""
echo "=== SUITE C2: Large-Scale Stress Test (Synthetic, N up to 10M, K=32) ==="
for n in 1000000 5000000 10000000; do
    echo "--- Suite C2 (Synthetic Stress): N=${n} ---"
    for cuda_impl in kmeans_cuda_naive kmeans_cuda_shmem kmeans_cuda_device; do
        ./bin/${cuda_impl} --synthetic --n ${n} --d 18 --k 32 --iters 20 --block-size 256 --warmup 1 --trials 3 --csv "${SUITE_C}"
    done
done

# -----------------------------------------------------------------------------
echo ""
echo "=== SUITE D: Cluster Count K Sweep (N=1M, D=18, K in 8..1024) ==="
for k in 8 16 32 64 128 256 512 1024; do
    echo "--- Suite D: K=${k} ---"
    if [ ${k} -le 256 ]; then
        ./bin/kmeans_seq --file "${SUSY_BIN}" --n 1000000 --k ${k} --iters 20 --warmup 0 --trials 2 --csv "${SUITE_D}"
    fi
    for cuda_impl in kmeans_cuda_naive kmeans_cuda_shmem kmeans_cuda_device; do
        ./bin/${cuda_impl} --file "${SUSY_BIN}" --n 1000000 --k ${k} --iters 20 --block-size 256 --warmup 1 --trials 5 --csv "${SUITE_D}"
    done
done

# -----------------------------------------------------------------------------
echo ""
echo "=== SUITE E1: CUDA Thread Block Size Sweep (N=5M, D=18, K=32) ==="
for bs in 64 128 256 512 1024; do
    echo "--- Suite E1 (Block Size): block_size=${bs} ---"
    for cuda_impl in kmeans_cuda_naive kmeans_cuda_shmem kmeans_cuda_device; do
        ./bin/${cuda_impl} --file "${SUSY_BIN}" --n 5000000 --k 32 --iters 20 --block-size ${bs} --warmup 1 --trials 5 --csv "${SUITE_E}"
    done
done

echo ""
echo "=== SUITE E2: Dimensionality D Sweep (N=1M, K=32, Synthetic) ==="
for d in 8 16 32 64; do
    echo "--- Suite E2 (Dimensionality): D=${d} ---"
    ./bin/kmeans_seq --synthetic --n 1000000 --d ${d} --k 32 --iters 20 --warmup 0 --trials 2 --csv "${SUITE_E}"
    for cuda_impl in kmeans_cuda_naive kmeans_cuda_shmem kmeans_cuda_device; do
        ./bin/${cuda_impl} --synthetic --n 1000000 --d ${d} --k 32 --iters 20 --block-size 256 --warmup 1 --trials 5 --csv "${SUITE_E}"
    done
done

# -----------------------------------------------------------------------------
echo ""
echo "=== 5M VERIFICATION GATE: kmeans_cuda_device on SUSY 5M ==="
./bin/kmeans_cuda_device --file "${SUSY_BIN}" --n 5000000 --k 32 --iters 20 --block-size 256 --verify | tee "${CHECK_5M}"

echo ""
echo "================================================================================"
echo "ALL BENCHMARK SUITES COMPLETED SUCCESSFULLY!"
echo "CSVs written: ${SUITE_B}, ${SUITE_C}, ${SUITE_D}, ${SUITE_E}"
echo "5M Gate check: ${CHECK_5M}"
echo "================================================================================"
