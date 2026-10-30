#!/bin/bash
# ==============================================================================
# sync_arc.sh - Synchronization Helper for NC State ARC Cluster
# Usage:
#   bash sync_arc.sh push <unityid> [--with-matrices]
#   bash sync_arc.sh pull <unityid>
# ==============================================================================

set -e

ACTION="$1"
UNITY_ID="${2:-fclin}"
EXTRA_FLAG="$3"

ARC_HOST="login.arc.ncsu.edu"
REMOTE_PATH="~/CSC548"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -z "$ACTION" ] || [ "$ACTION" == "--help" ] || [ "$ACTION" == "-h" ]; then
    echo "=================================================================="
    echo " NC State ARC Sync Utility for CSC 548 HW3"
    echo "=================================================================="
    echo "Usage:"
    echo "  bash sync_arc.sh push [unityid] [--with-matrices]"
    echo "      Uploads HW3 code, kernels, and scripts to ARC cluster."
    echo "      Pass --with-matrices to also upload the 222 MB matrix suite."
    echo ""
    echo "  bash sync_arc.sh pull [unityid]"
    echo "      Downloads benchmark CSVs, plots, report.pdf, and HW3-fclin.zip"
    echo "      from ARC back to your local repository."
    echo "=================================================================="
    exit 0
fi

cd "$REPO_ROOT"

if [ "$ACTION" == "push" ]; then
    echo "=================================================================="
    echo " Pushing HW3 code and scripts to ARC ($UNITY_ID@$ARC_HOST)..."
    echo "=================================================================="
    
    ssh "$UNITY_ID@$ARC_HOST" "mkdir -p $REMOTE_PATH/hw3/SpMV/matrices"
    
    rsync -avz --progress \
        --exclude '.git' \
        --exclude 'venv' \
        --exclude 'venv_arc' \
        --exclude '__pycache__' \
        --exclude '*.o' \
        --exclude 'spmv' \
        --exclude 'spmv_csr' \
        --exclude 'spmv_cuda_*' \
        --exclude 'matrices' \
        hw3/ "$UNITY_ID@$ARC_HOST:$REMOTE_PATH/hw3/"

    if [ "$EXTRA_FLAG" == "--with-matrices" ]; then
        echo ""
        echo "Pushing matrix suite to ARC (approx. 222 MB)..."
        if [ -d "hw2/SpMV/matrices" ]; then
            rsync -avz --progress hw2/SpMV/matrices/ "$UNITY_ID@$ARC_HOST:$REMOTE_PATH/hw3/SpMV/matrices/"
        elif [ -d "hw3/SpMV/matrices" ]; then
            rsync -avz -L --progress hw3/SpMV/matrices/ "$UNITY_ID@$ARC_HOST:$REMOTE_PATH/hw3/SpMV/matrices/"
        fi
    fi

    echo ""
    echo "[SUCCESS] Code pushed to ARC successfully!"
    echo "Next steps on ARC:"
    echo "  ssh $UNITY_ID@$ARC_HOST"
    echo "  cd $REMOTE_PATH/hw3/SpMV"
    echo "  sbatch run_benchmarks.batch"

elif [ "$ACTION" == "pull" ]; then
    echo "=================================================================="
    echo " Pulling HW3 results & deliverables from ARC ($UNITY_ID@$ARC_HOST)..."
    echo "=================================================================="

    mkdir -p hw3/SpMV/results/plots
    mkdir -p hw3/HW3-$UNITY_ID

    rsync -avz --progress "$UNITY_ID@$ARC_HOST:$REMOTE_PATH/hw3/SpMV/results/" hw3/SpMV/results/
    rsync -avz --progress "$UNITY_ID@$ARC_HOST:$REMOTE_PATH/hw3/report.pdf" hw3/report.pdf || true
    rsync -avz --progress "$UNITY_ID@$ARC_HOST:$REMOTE_PATH/hw3/HW3-$UNITY_ID.zip" hw3/ || true
    rsync -avz --progress "$UNITY_ID@$ARC_HOST:$REMOTE_PATH/hw3/HW3-$UNITY_ID/" hw3/HW3-$UNITY_ID/ || true

    echo ""
    echo "[SUCCESS] Results and deliverables pulled from ARC successfully!"
    echo "Local files updated:"
    echo "  - hw3/HW3-$UNITY_ID.zip"
    echo "  - hw3/report.pdf"
    echo "  - hw3/SpMV/results/all_benchmarks_combined.csv"

else
    echo "[ERROR] Unknown action '$ACTION'. Use 'push' or 'pull'."
    exit 1
fi

