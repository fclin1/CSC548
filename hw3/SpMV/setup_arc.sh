#!/bin/bash
# ==============================================================================
# setup_arc.sh - Environment Setup & Verification for NC State ARC GPU Nodes
# Course: CSC 548 Parallel Systems
# Target Hardware: NVIDIA Tesla V100 (Volta sm_70)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo " Setting up HW3 CUDA Environment on ARC GPU Node: $(hostname)"
echo " Date: $(date)"
echo " Working Directory: $SCRIPT_DIR"
echo "=================================================================="

# ------------------------------------------------------------------------------
# 1. Module Management
# ------------------------------------------------------------------------------
echo ""
echo "[Step 1/5] Checking Environment Modules..."
if command -v module &> /dev/null; then
    echo "Module command found. Loading CUDA, GCC, and Python modules..."
    # Attempt loading standard ARC module variations
    module load cuda/11.8.0 2>/dev/null || module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || echo "Note: Using default or system CUDA"
    module load gcc/10.2.0 2>/dev/null || module load gcc 2>/dev/null || echo "Note: Using default GCC"
    module load python/3.10 2>/dev/null || module load python/3.8 2>/dev/null || module load python 2>/dev/null || echo "Note: Using system Python3"
else
    echo "Note: 'module' command not available or already in compute environment."
fi

# Verify nvcc
echo "Verifying nvcc compiler..."
if ! command -v nvcc &> /dev/null; then
    echo "[ERROR] nvcc compiler not found in PATH!"
    echo "Please run: 'module load cuda' before executing."
    exit 1
fi
echo "Found nvcc: $(which nvcc)"
nvcc --version | grep "release" || true

# Verify nvidia-smi
echo "Verifying NVIDIA GPU access..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || nvidia-smi
else
    echo "[WARNING] nvidia-smi not available on this node (are you on a login node instead of a GPU compute node?)."
fi

# ------------------------------------------------------------------------------
# 2. Benchmark Matrices Verification
# ------------------------------------------------------------------------------
echo ""
echo "[Step 2/5] Checking Benchmark Matrix Suite..."
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
    echo "Matrix suite not detected in 'matrices/'. Searching candidate locations..."
    
    CANDIDATES=(
        "$SCRIPT_DIR/../../hw2/SpMV/matrices"
        "$SCRIPT_DIR/../../hw1/SpMV/matrices"
        "$SCRIPT_DIR/../matrices"
        "$HOME/CSC548/hw2/SpMV/matrices"
        "/share/$USER/matrices"
    )

    found_dir=""
    for cand in "${CANDIDATES[@]}"; do
        if [ -d "$cand" ] && [ -f "$cand/D6-6.mtx" ] && [ -f "$cand/pkustk14.mtx" ]; then
            found_dir="$cand"
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
        echo "  rsync -avz hw2/SpMV/matrices/ <unityid>@login.arc.ncsu.edu:~/CSC548/hw3/SpMV/matrices/"
        echo "Or copy them to: $SCRIPT_DIR/matrices/"
        if [ "$1" != "--check-only" ]; then
            exit 1
        fi
    fi
else
    echo "All 6 benchmark matrices are present in matrices/."
fi

# ------------------------------------------------------------------------------
# 3. Python Virtual Environment Setup
# ------------------------------------------------------------------------------
echo ""
echo "[Step 3/5] Setting up Python Virtual Environment (venv_arc)..."
if [ ! -d "venv_arc" ]; then
    echo "Creating virtual environment 'venv_arc' using python3..."
    python3 -m venv venv_arc
    echo "Installing required report packages (matplotlib, pandas, reportlab, numpy)..."
    venv_arc/bin/pip install --no-cache-dir matplotlib pandas reportlab numpy
    echo "Virtual environment successfully configured."
else
    echo "'venv_arc' already exists."
    # Verify reportlab is installed
    if ! venv_arc/bin/python3 -c "import reportlab" &> /dev/null; then
        echo "Installing missing dependencies into venv_arc..."
        venv_arc/bin/pip install --no-cache-dir matplotlib pandas reportlab numpy
    fi
fi

# Quick test of venv
venv_arc/bin/python3 -c "import matplotlib, pandas, reportlab, numpy; print('Python packages verified: OK')"

# ------------------------------------------------------------------------------
# 4. Compilation & Architecture Build
# ------------------------------------------------------------------------------
echo ""
echo "[Step 4/5] Compiling CUDA Kernels for Volta sm_70..."
make clean
make all GPU_ARCH=sm_70
echo "Build succeeded. Generated binaries:"
ls -lh spmv spmv_csr spmv_cuda_scalar spmv_cuda_vector spmv_cuda_ell

# ------------------------------------------------------------------------------
# 5. Sanity Test Execution
# ------------------------------------------------------------------------------
echo ""
echo "[Step 5/5] Running Smoke Test Verification on matrices/bfly.mtx..."
if [ -f "matrices/bfly.mtx" ] && command -v nvidia-smi &> /dev/null; then
    echo "Testing CSR Scalar..."
    ./spmv_cuda_scalar matrices/bfly.mtx --block=256
    echo "Testing CSR Vector..."
    ./spmv_cuda_vector matrices/bfly.mtx --block=256
    echo "Testing ELLPACK..."
    ./spmv_cuda_ell matrices/bfly.mtx --block=256
    echo "All smoke tests PASSED."
else
    echo "Note: Smoke test skipped (not on active GPU node or bfly.mtx missing)."
fi

echo ""
echo "=================================================================="
echo " [SUCCESS] ARC GPU Environment is Fully Ready!"
echo " Next Steps:"
echo "   - Submit Batch Job:    sbatch run_benchmarks.batch"
echo "   - Or Run Interactive:  bash run_experiments.sh"
echo "=================================================================="

