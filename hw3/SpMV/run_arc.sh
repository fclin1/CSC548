#!/bin/bash
# ==============================================================================
# run_arc.sh - Minimal Benchmark Runner on NC State ARC GPU Node
# Generates all benchmark raw data (CSV + CPU baselines + hardware topology)
# ==============================================================================
set -e
cd "$(dirname "$0")"

echo "=========================================================="
echo "Starting HW3 Benchmark Run on ARC: $(hostname)"
echo "Date: $(date)"
echo "=========================================================="

# 1. Load environment modules if available
if command -v module &> /dev/null; then
    module load cuda gcc 2>/dev/null || true
fi

# 2. Check matrices (link from HW2 if available)
if [ ! -d "matrices" ]; then
    if [ -d "../../hw2/SpMV/matrices" ]; then
        echo "Symlinking matrices from hw2..."
        ln -s ../../hw2/SpMV/matrices matrices
    elif [ -d "../matrices" ]; then
        echo "Symlinking matrices from parent directory..."
        ln -s ../matrices matrices
    fi
fi

if [ ! -d "matrices" ] || [ ! -f "matrices/bfly.mtx" ]; then
    echo "ERROR: 'matrices/' directory not found or incomplete!"
    echo "Copy matrices from local: scp -r hw2/SpMV/matrices <unityid>@login.arc.ncsu.edu:~/CSC548/hw3/SpMV/"
    exit 1
fi


# 3. Build all kernels (sm_120)
echo "Building all kernels with GPU_ARCH=sm_120..."
make clean
make all GPU_ARCH=sm_120

# 4. Run CPU and GPU benchmarks (generates all results)
echo "Executing benchmarks..."
bash run_experiments.sh

echo "=========================================================="
echo "SUCCESS! Results generated in: hw3/SpMV/results/"
echo "Files ready to pull:"
ls -lh results/all_benchmarks_combined.csv results/cpu_baselines.txt results/hardware_info.txt
echo "=========================================================="

