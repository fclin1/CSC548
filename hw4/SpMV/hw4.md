# [HW4] Spark — SpMV (100 points)

**Release:** Mon, Nov 2, 2026  
**Due:** Wed, Nov 11, 2026 11:59pm ET

**Single-person assignment.**

## Introduction

Finally we run **SpMV in Apache Spark** — and the goal here is as much about **understanding the model's cost** as about the code. Spark shines at large-scale data-parallel ETL, not tight numerical kernels, so this assignment also asks you to explain *why* SpMV is slow in Spark relative to MPI/CUDA.

Represent the matrix as an RDD/DataFrame of `(row, col, val)` and `x` as a small broadcast/co-partitioned collection. Datasets: SuiteSparse **D6-6, dictionary28, Ga3As3H12, bfly, pkustk14, roadNet-CA**. You may also test on **other SuiteSparse matrices** of your choice — but you must **explain your reason** in the report (e.g. to probe a different size, density, or row-length-variance regime). Run with `spark-submit --master 'local[*]'` (or on ARC's Spark module); your laptop is fine if ARC's Spark is hard to set up.

## Tasks

You implement SpMV in Spark and make its partitioning controllable (Tasks 1–2), then write them up in a **report** whose required contents are graded separately (Tasks 1–3 under Report).

### Code

**1. SpMV in Spark (40 pts).** Implement `spmv-spark.py`: join/aggregate `(row, col, val)` with `x` to produce `y` (e.g. `map` to `(col, val)`, join with `x`, `map` to `(row, val*x)`, `reduceByKey`). Verify against the sequential baseline.

**2. Partitioning (15 pts).** Make the number of partitions of `A` a command-line parameter (`parallelize(..., numSlices)` or `repartition`). Implement two ways of bringing `x` to the nonzeros of `A`: (a) the plain `join` on the column key, and (b) a **broadcast** of `x` or **co-partitioning** `x` with `A` under the same partitioner, which removes the join's shuffle. Print the lineage with `toDebugString` so the shuffle stages are visible.

### Report

Back every analysis with figures and/or tables — a claim without a supporting plot or table earns little credit.

**1. Partitioning & shuffle study (15 pts).** Runtime vs number of partitions (e.g. 1, 2, 4, 8, 16, 32) for the plain join vs the broadcast/co-partitioned version; shuffle read/write bytes and number of stages from the Spark UI or `toDebugString`. Explain where each shuffle comes from (the `join` on `col`, the `reduceByKey` on `row`) and which one co-partitioning removes.

**2. Performance & cross-model analysis (25 pts).** From your measurements:

- Runtime vs number of partitions/executor cores (a strong-scaling-style plot) and the resulting **speedup/efficiency**.
- **Compare to your MPI / OpenMP / CUDA results** from HW1–HW3 on the same matrices.
- **Key question:** Spark SpMV is typically **10–100× slower** than MPI/CUDA. Explain why, using your data — lineage/DAG scheduling, **shuffle**, JVM/Python serialization overhead, and the fact that Spark targets data-parallel ETL, not fine-grained numeric kernels.

**3. Write-up (5 pts).** Implementation design, per-matrix results, and the cross-model cost discussion.

## Submission

Folder `HW4-unityID/` with `spmv-spark.py`, plots, and `report.pdf` (plots included); zip to `HW4-unityID.zip`.

## Grading

- Spark SpMV (correctness): **40**
- Partitioning (code): **15**
- Partitioning & shuffle study: **15**
- Performance & cross-model analysis (speedup + the "why slow" reasoning): **25**
- Write-up: **5**
