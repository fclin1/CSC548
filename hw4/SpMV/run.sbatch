#!/bin/bash
#SBATCH --job-name=hw4
#SBATCH --partition=class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# First run: both schemes on one matrix.
#   sbatch run.sbatch
#   MATRIX=matrices/bfly.mtx PARTITIONS=8 sbatch run.sbatch
#   SCHEMES="join copartition" sbatch run.sbatch

set -e

# ARC compute-node environment; exported values select a local installation.
export SPARK_HOME="${SPARK_HOME:-/usr/local/spark}"
export PATH="$SPARK_HOME/bin:$PATH"
# Use Python 3.11 for both the driver and workers.
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-/usr/bin/python3.11}"
export PYSPARK_DRIVER_PYTHON="${PYSPARK_DRIVER_PYTHON:-$PYSPARK_PYTHON}"

for scheme in ${SCHEMES:-join broadcast}; do
  spark-submit --master "local[${SLURM_CPUS_PER_TASK:-*}]" --driver-memory "${DRIVER_MEMORY:-16g}" \
    spmv-spark.py "${MATRIX:-matrices/D6-6.mtx}" --scheme "$scheme" --partitions "${PARTITIONS:-4}"
done
