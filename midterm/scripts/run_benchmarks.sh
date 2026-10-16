#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEMM_DIR="$DIR/GEMM"
RESULTS_DIR="$DIR/results"

mkdir -p "$RESULTS_DIR"

echo "=== 0. Recording System Hardware Specs ==="
lscpu > "$RESULTS_DIR/hardware_info.txt"

echo "=== 1. Measuring STREAM Memory Bandwidth Ceiling ==="
cd "$GEMM_DIR"
make stream
OMP_NUM_THREADS=6 ./stream > "$RESULTS_DIR/stream_bandwidth.txt"
echo "--- 1-thread STREAM ---" >> "$RESULTS_DIR/stream_bandwidth.txt"
OMP_NUM_THREADS=1 ./stream 32000000 >> "$RESULTS_DIR/stream_bandwidth.txt"

echo "=== 2. Compiling Binaries ==="
make clean
make all

echo "=== 3. Profiling Naive GEMM (perf & gprof) ==="
# perf stat on gemm_naive
perf stat -e cycles,instructions,cache-misses,cache-references \
    ./gemm_naive --n=512 --iters=1 --no-check 2> "$RESULTS_DIR/perf_naive_512.txt"
perf stat -e cycles,instructions,cache-misses,cache-references \
    ./gemm_naive --n=1024 --iters=1 --no-check 2> "$RESULTS_DIR/perf_naive_1024.txt"

# perf stat on gemm_tiled for comparison
perf stat -e cycles,instructions,cache-misses,cache-references \
    ./gemm_tiled --n=512 --tile=64 --iters=1 --no-check 2> "$RESULTS_DIR/perf_tiled_512.txt"
perf stat -e cycles,instructions,cache-misses,cache-references \
    ./gemm_tiled --n=1024 --tile=64 --iters=1 --no-check 2> "$RESULTS_DIR/perf_tiled_1024.txt"

# gprof profiling on naive
make clean
make gemm_naive PROFILE=1
./gemm_naive --n=512 --iters=1 --no-check
gprof -b gemm_naive > "$RESULTS_DIR/gprof_naive.txt"

# Rebuild normal binaries
make clean
make all

echo "=== 4. Benchmarking Naive Baseline ==="
./gemm_naive --csv-header > "$RESULTS_DIR/baseline_naive.csv"
for N in 512 1024 2048; do
    echo "Running gemm_naive N=$N"
    ./gemm_naive --n=$N --csv >> "$RESULTS_DIR/baseline_naive.csv"
done

echo "=== 5. Part 1b: Tiling Sweeps ==="
./gemm_tiled --csv-header > "$RESULTS_DIR/sweep_tiled.csv"
for N in 512 1024 2048; do
    for T in 16 32 64 128 256; do
        echo "Running gemm_tiled N=$N tile=$T"
        ./gemm_tiled --n=$N --tile=$T --csv >> "$RESULTS_DIR/sweep_tiled.csv"
    done
done

echo "=== 6. Part 1c: OpenMP Strong Scaling ==="
./gemm_omp --csv-header > "$RESULTS_DIR/scaling_omp.csv"
for N in 1024 2048 4096; do
    for TH in 1 2 3 4 5 6; do
        echo "Running gemm_omp N=$N threads=$TH"
        ./gemm_omp --n=$N --threads=$TH --tile=64 --csv >> "$RESULTS_DIR/scaling_omp.csv"
    done
done

echo "=== 7. Part 1d: SIMD Vectorization Sweeps ==="
./gemm_simd --csv-header > "$RESULTS_DIR/scaling_simd.csv"
for N in 512 1024 2048 4096; do
    for TH in 1 6; do
        echo "Running gemm_simd N=$N threads=$TH"
        ./gemm_simd --n=$N --threads=$TH --tile=64 --csv >> "$RESULTS_DIR/scaling_simd.csv"
    done
done

echo "=== 8. Part 2a/2b: MPI SUMMA Strong Scaling ==="
./gemm_mpi --csv-header > "$RESULTS_DIR/summa_strong_scaling.csv"
for N in 1536 3072; do
    for P in 4 9 16; do
        echo "Running gemm_mpi Strong Scaling N=$N P=$P"
        mpirun --oversubscribe -n $P --bind-to none ./gemm_mpi --n=$N --threads=1 --tile=64 --csv >> "$RESULTS_DIR/summa_strong_scaling.csv"
    done
done

echo "=== 9. Part 2b: MPI SUMMA Weak Scaling (b=768 fixed) ==="
./gemm_mpi --csv-header > "$RESULTS_DIR/summa_weak_scaling.csv"
# b=768: P=4 (p=2) -> N=1536; P=9 (p=3) -> N=2304; P=16 (p=4) -> N=3072
echo "Running gemm_mpi Weak Scaling P=4 N=1536 (b=768)"
mpirun --oversubscribe -n 4 --bind-to none ./gemm_mpi --n=1536 --threads=1 --tile=64 --csv >> "$RESULTS_DIR/summa_weak_scaling.csv"

echo "Running gemm_mpi Weak Scaling P=9 N=2304 (b=768)"
mpirun --oversubscribe -n 9 --bind-to none ./gemm_mpi --n=2304 --threads=1 --tile=64 --csv >> "$RESULTS_DIR/summa_weak_scaling.csv"

echo "Running gemm_mpi Weak Scaling P=16 N=3072 (b=768)"
mpirun --oversubscribe -n 16 --bind-to none ./gemm_mpi --n=3072 --threads=1 --tile=64 --csv >> "$RESULTS_DIR/summa_weak_scaling.csv"

echo "=== All Benchmarks Completed Successfully! ==="

