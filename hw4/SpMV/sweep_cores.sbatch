#!/bin/bash
#SBATCH --job-name=hw4-cores
#SBATCH --partition=class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Executor Core Strong-Scaling Sweep at fixed partition count (p=16).
# Results are appended to results/<jobid>/ or results/cores/.

export SPARK_HOME="${SPARK_HOME:-/usr/local/spark}"
export PATH="$SPARK_HOME/bin:$PATH"
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-/usr/bin/python3.11}"
export PYSPARK_DRIVER_PYTHON="${PYSPARK_DRIVER_PYTHON:-$PYSPARK_PYTHON}"

OUT_DIR="${OUT_DIR:-results/${SLURM_JOB_ID:-cores}}"
mkdir -p "$OUT_DIR"
status=0
PARTITIONS="${PARTITIONS:-16}"

echo "Starting Executor Core Scaling Sweep (cores: 1 2 4 8 16, partitions: $PARTITIONS)..."

for m in ${MATRICES:-D6-6 pkustk14 roadNet-CA}; do
  for scheme in ${SCHEMES:-broadcast join}; do
    for c in ${CORES:-1 2 4 8 16}; do
      name="$m-$scheme-c$c-p$PARTITIONS"
      echo "Running $name..."
      spark-submit --master "local[$c]" --driver-memory "${DRIVER_MEMORY:-16g}" \
        spmv-spark.py "matrices/$m.mtx" --scheme "$scheme" --partitions "$PARTITIONS" --iters 5 \
        --csv "$OUT_DIR/spark_cores.csv" --event-log-dir "$OUT_DIR/event-logs-cores" --json "$OUT_DIR/$name.json" \
        > "$OUT_DIR/$name.log" 2>&1 || {
          echo "FAILED: $name (see $OUT_DIR/$name.log)"
          status=1
        }
    done
  done
done

"$PYSPARK_PYTHON" tools/event_metrics.py "$OUT_DIR/event-logs-cores" > "$OUT_DIR/shuffle_cores.csv" || status=1
echo "Core scaling sweep complete. Results written to $OUT_DIR."
exit "$status"
