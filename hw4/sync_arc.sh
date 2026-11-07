#!/bin/bash
# ==============================================================================
# sync_arc.sh - Synchronization Helper for NC State ARC Cluster (HW4 Spark SpMV)
# Usage:
#   bash sync_arc.sh push [unityid] [--with-matrices]
#   bash sync_arc.sh pull [unityid]
#   bash sync_arc.sh status [unityid]
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
    echo " NC State ARC Sync Utility for CSC 548 HW4 (Spark SpMV)"
    echo "=================================================================="
    echo "Usage:"
    echo "  bash sync_arc.sh push [unityid] [--with-matrices]"
    echo "      Uploads HW4 code, scripts, and tools to ARC cluster."
    echo "      Pass --with-matrices to also upload the matrices if needed."
    echo ""
    echo "  bash sync_arc.sh pull [unityid]"
    echo "      Downloads benchmark results (spark.csv, shuffle.csv, event-logs),"
    echo "      plots, report.pdf, and HW4-fclin.zip from ARC back to local repo."
    echo ""
    echo "  bash sync_arc.sh status [unityid]"
    echo "      Checks active Slurm jobs on ARC for the user."
    echo "=================================================================="
    exit 0
fi

cd "$REPO_ROOT"

if [ "$ACTION" == "push" ]; then
    echo "=================================================================="
    echo " Pushing HW4 code and scripts to ARC ($UNITY_ID@$ARC_HOST)..."
    echo "=================================================================="

    ssh "$UNITY_ID@$ARC_HOST" "mkdir -p $REMOTE_PATH/hw4/SpMV/matrices"

    rsync -avz --progress \
        --exclude '.git' \
        --exclude 'venv' \
        --exclude 'venv_arc' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude 'results' \
        --exclude 'matrices' \
        hw4/ "$UNITY_ID@$ARC_HOST:$REMOTE_PATH/hw4/"

    # Link matrices on ARC if already present from previous homeworks
    ssh "$UNITY_ID@$ARC_HOST" "bash -c '
        if [ ! -f $REMOTE_PATH/hw4/SpMV/matrices/D6-6.mtx ]; then
            for prev in hw3 hw2 hw1; do
                if [ -f $REMOTE_PATH/\$prev/SpMV/matrices/D6-6.mtx ]; then
                    echo \"Linking matrices from \$prev/SpMV/matrices/...\";
                    ln -sf $REMOTE_PATH/\$prev/SpMV/matrices/* $REMOTE_PATH/hw4/SpMV/matrices/;
                    break;
                fi
            done
        fi
    '"

    if [ "$EXTRA_FLAG" == "--with-matrices" ]; then
        echo ""
        echo "Pushing matrix suite to ARC from local hw4/SpMV/matrices/..."
        rsync -avz --progress hw4/SpMV/matrices/ "$UNITY_ID@$ARC_HOST:$REMOTE_PATH/hw4/SpMV/matrices/"
    fi

    echo ""
    echo "[SUCCESS] HW4 code pushed to ARC successfully!"
    echo "Next steps on ARC:"
    echo "  ssh $UNITY_ID@$ARC_HOST"
    echo "  cd $REMOTE_PATH/hw4/SpMV"
    echo "  sbatch run.sbatch     # Quick smoke test"
    echo "  sbatch sweep.sbatch   # Full 72-configuration sweep"

elif [ "$ACTION" == "pull" ]; then
    echo "=================================================================="
    echo " Pulling HW4 results & deliverables from ARC ($UNITY_ID@$ARC_HOST)..."
    echo "=================================================================="

    mkdir -p hw4/SpMV/results/plots
    mkdir -p hw4/HW4-$UNITY_ID

    rsync -avz --progress "$UNITY_ID@$ARC_HOST:$REMOTE_PATH/hw4/SpMV/results/" hw4/SpMV/results/ || true
    rsync -avz --progress "$UNITY_ID@$ARC_HOST:$REMOTE_PATH/hw4/report.pdf" hw4/report.pdf || true
    rsync -avz --progress "$UNITY_ID@$ARC_HOST:$REMOTE_PATH/hw4/HW4-$UNITY_ID.zip" hw4/ || true
    rsync -avz --progress "$UNITY_ID@$ARC_HOST:$REMOTE_PATH/hw4/HW4-$UNITY_ID/" hw4/HW4-$UNITY_ID/ || true

    echo ""
    echo "[SUCCESS] Results and deliverables pulled from ARC successfully!"

elif [ "$ACTION" == "status" ]; then
    echo "=================================================================="
    echo " Slurm Queue Status on ARC for $UNITY_ID..."
    echo "=================================================================="
    ssh "$UNITY_ID@$ARC_HOST" "squeue -u $UNITY_ID"

else
    echo "[ERROR] Unknown action '$ACTION'. Use 'push', 'pull', or 'status'."
    exit 1
fi
