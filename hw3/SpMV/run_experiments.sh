#!/bin/bash
set -e

cd "$(dirname "$0")"

mkdir -p results/plots

echo "=== Recording hardware info ==="
lscpu > results/hardware_info.txt
nvidia-smi >> results/hardware_info.txt

MATRICES=(
    "matrices/D6-6.mtx"
    "matrices/dictionary28.mtx"
    "matrices/bfly.mtx"
    "matrices/Ga3As3H12.mtx"
    "matrices/pkustk14.mtx"
    "matrices/roadNet-CA.mtx"
)

BLOCK_SIZES=(64 128 256 512 1024)

CSV_HEADER=$(./spmv_cuda_scalar --csv-header)

echo "$CSV_HEADER,kernel" > results/all_benchmarks_combined.csv
echo "$CSV_HEADER" > results/scalar_block_sweep.csv
echo "$CSV_HEADER" > results/vector_block_sweep.csv
echo "$CSV_HEADER" > results/ell_block_sweep.csv

echo "=== Running CPU Baselines ==="
> results/cpu_baselines.txt
for mat in "${MATRICES[@]}"; do
    echo "Running spmv_csr on $mat..."
    ./spmv_csr "$mat" >> results/cpu_baselines.txt 2>&1
done

echo "=== Running GPU Benchmarks ==="
for b in "${BLOCK_SIZES[@]}"; do
    echo "--- Block size: $b ---"
    for mat in "${MATRICES[@]}"; do
        echo "Running scalar: $mat (block=$b)..."
        line_scalar=$(./spmv_cuda_scalar "$mat" --block="$b" --csv)
        echo "$line_scalar" >> results/scalar_block_sweep.csv
        echo "$line_scalar,scalar" >> results/all_benchmarks_combined.csv

        echo "Running vector: $mat (block=$b)..."
        line_vector=$(./spmv_cuda_vector "$mat" --block="$b" --csv)
        echo "$line_vector" >> results/vector_block_sweep.csv
        echo "$line_vector,vector" >> results/all_benchmarks_combined.csv

        echo "Running ell: $mat (block=$b)..."
        line_ell=$(./spmv_cuda_ell "$mat" --block="$b" --csv)
        echo "$line_ell" >> results/ell_block_sweep.csv
        echo "$line_ell,ell" >> results/all_benchmarks_combined.csv
    done
done

echo "=== All benchmarks completed successfully! ==="

