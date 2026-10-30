#!/bin/bash
# ==============================================================================
# run_interactive.sh - Interactive Benchmark Runner for ARC GPU Sessions
# Usage:
#   1. Request an interactive GPU session:
#      srun -p gpu --gres=gpu:1 -c 4 -t 01:00:00 --pty bash -i
#   2. Run this script:
#      bash run_interactive.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo " CSC 548 HW3: Interactive GPU Benchmark Runner"
echo " Host: $(hostname)"
echo " Date: $(date)"
echo "=================================================================="

# 1. Environment & Setup
bash setup_arc.sh

# 2. Benchmark Execution
echo ""
echo "=== Running Benchmark Suite ==="
bash run_experiments.sh

# 3. Post-Processing & Report Compilation
echo ""
echo "=== Generating Visualizations and Report ==="
if [ -d "venv_arc" ]; then
    PYTHON_CMD="venv_arc/bin/python3"
else
    PYTHON_CMD="python3"
fi

$PYTHON_CMD generate_plots.py
$PYTHON_CMD build_report.py

# 4. Package deliverables
echo ""
echo "=== Updating Submission Deliverables ==="
mkdir -p ../HW3-fclin
cp src/spmv-cuda-scalar.cu ../HW3-fclin/
cp src/spmv-cuda-vector.cu ../HW3-fclin/
cp src/spmv-cuda-ell.cu    ../HW3-fclin/
cp Makefile                ../HW3-fclin/
cp report.pdf              ../HW3-fclin/
cp report.pdf              ../report.pdf

cd ..
rm -f HW3-fclin.zip
zip -r HW3-fclin.zip HW3-fclin/
cd "$SCRIPT_DIR"

echo ""
echo "=================================================================="
echo " [SUCCESS] Interactive Benchmarking Complete!"
echo " Outputs ready for retrieval:"
echo "   - hw3/HW3-fclin.zip"
echo "   - hw3/report.pdf"
echo "   - hw3/SpMV/results/all_benchmarks_combined.csv"
echo "=================================================================="

