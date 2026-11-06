#!/bin/bash
#SBATCH --job-name=hw4-sweep
#SBATCH --partition=class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Partition sweep at a fixed core count; results go to results/<jobid>/.
#   sbatch sweep.sbatch
#   CORES=8 sbatch sweep.sbatch
#   MATRICES="D6-6 bfly" PARTITIONS="1 4" sbatch sweep.sbatch
#   SCHEMES="join copartition" sbatch sweep.sbatch

# ARC compute-node environment; exported values select a local installation.
export SPARK_HOME="${SPARK_HOME:-/usr/local/spark}"
export PATH="$SPARK_HOME/bin:$PATH"
# Use Python 3.11 for both the driver and workers.
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-/usr/bin/python3.11}"
export PYSPARK_DRIVER_PYTHON="${PYSPARK_DRIVER_PYTHON:-$PYSPARK_PYTHON}"

CORES="${CORES:-4}"
OUT_DIR="${OUT_DIR:-results/${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT_DIR"
status=0
for m in ${MATRICES:-D6-6 dictionary28 Ga3As3H12 bfly pkustk14 roadNet-CA}; do
  for scheme in ${SCHEMES:-join broadcast}; do
    for p in ${PARTITIONS:-1 2 4 8 16 32}; do
      name="$m-$scheme-c$CORES-p$p"
      spark-submit --master "local[$CORES]" --driver-memory "${DRIVER_MEMORY:-16g}" \
        spmv-spark.py "matrices/$m.mtx" --scheme "$scheme" --partitions "$p" --iters 5 \
        --csv "$OUT_DIR/spark.csv" --event-log-dir "$OUT_DIR/event-logs" --json "$OUT_DIR/$name.json" \
        > "$OUT_DIR/$name.log" 2>&1 || {
          echo "FAILED: $name (see $OUT_DIR/$name.log)"
          status=1
        }
    done
  done
done
"$PYSPARK_PYTHON" tools/event_metrics.py "$OUT_DIR/event-logs" > "$OUT_DIR/shuffle.csv" || status=1
exit "$status"
