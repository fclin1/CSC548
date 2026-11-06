# HW4 starter — Spark SpMV

Implement Spark SpMV, measure partitioning and communication costs, and compare
your results with HW1–HW3. [hw4.md](hw4.md) gives the tasks and grading rubric.
Run the commands below from the unpacked `SpMV/` directory.

## Implement the three functions

Fill in the TODOs at the top of `spmv-spark.py`:

| Function | Your implementation |
| --- | --- |
| `make_matrix_rdd(sc, entries, partitions)` | Create an RDD of `(row, col, val)` with the requested number of partitions. |
| `spmv_join(a_rdd, x_rdd, partitions)` | Key A by column, retain `(row, val)`, join with x, and sum products by row. |
| `spmv_broadcast(a_rdd, x_broadcast, partitions)` | Read `x_broadcast.value` inside the worker function, multiply each nonzero by `x[col]`, and sum products by row. |

Both SpMV functions return a lazy RDD of unique `(row, y[row])` pairs with
zero-based indices. Compute the products on workers. The harness supplies zero
values for empty rows.

The rest of the file provides matrix input, command-line options, x generation,
broadcast creation, lineage printing, sequential verification, timing and output.
The harness materializes the input RDDs and collects the result of each SpMV.
The three functions are the entry points in this RDD starter. You may also adapt
the template for a DataFrame implementation, as allowed in the assignment.

### Co-partitioning option

You may implement co-partitioning as your second strategy. Replace
`spmv_broadcast` with your co-partitioned function and update the harness:

1. Add `copartition` to the `--scheme` choices.
2. During setup, key A by column, give A and x the same partitioner, and cache
   and materialize both RDDs. Preserve the partitioner through the join.
3. Call your function for `--scheme copartition`. Keep the other command-line
   options and output fields to use the supplied run and measurement scripts.

Select that pair of schemes in either run script with
`SCHEMES="join copartition"`. The event metrics identify input preparation as
`setup` and timed SpMVs as `measure-*`.

## Prepare the matrices

Use the six SuiteSparse matrices from HW1–HW3: **D6-6, dictionary28,
Ga3As3H12, bfly, pkustk14 and roadNet-CA**. Copy or link your existing files
into `SpMV/matrices/`. For a link, replace the path below with your matrix folder:

```sh
ln -s /path/to/your/HW1/SpMV/matrices matrices
```

The run scripts use files named `matrices/D6-6.mtx`,
`matrices/dictionary28.mtx`, and so on. Contact the TA for a copy of the matrices.

The harness replaces the matrix values and generates x with a seeded Python random
generator, in double precision. Every SpMV is checked against a sequential result,
with a relative L2 tolerance of `1e-8`.

## Run on ARC

ARC compute nodes provide Spark 3.5.0, Python 3.11 and Java 8. The supplied
Slurm scripts select `/usr/local/spark`, `/usr/bin/python3.11` and a 16 GB driver
heap. After completing the functions and preparing the matrices, submit:

```sh
sbatch run.sbatch
MATRIX=matrices/bfly.mtx PARTITIONS=8 sbatch run.sbatch
```

`run.sbatch` runs both schemes on D6-6 with four partitions by default, using
the CPUs allocated to the job. Its output is `hw4-<jobid>.out`, including
correctness, timing and lineage. Each scheme reports `check: PASS` for a correct
result. The script requests 16 CPUs and a one-hour time limit.

For a co-partitioned implementation:

```sh
SCHEMES="join copartition" sbatch run.sbatch
```

### Run on a laptop

Use Python 3.11 and Java 8, 11 or 17. Install the Python dependencies once:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export PYSPARK_PYTHON="$(command -v python)"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
export SPARK_HOME="$(python -c 'import pyspark; print(pyspark.__path__[0])')"
DRIVER_MEMORY=4g bash run.sbatch
```

This runs the D6-6 example with PySpark 3.5.7. `DRIVER_MEMORY` selects the Java
heap size.

## Run the experiments

The supplied partition sweep runs all six matrices with both schemes at
1, 2, 4, 8, 16 and 32 partitions. It uses four cores, one warmup and five timed
SpMVs per configuration. The Slurm allocation is 16 CPUs with a four-hour limit.

```sh
sbatch sweep.sbatch
CORES=8 sbatch sweep.sbatch
MATRICES="D6-6 bfly" PARTITIONS="1 4" sbatch sweep.sbatch
SCHEMES="join copartition" sbatch sweep.sbatch
```

Use the first three commands for the broadcast template and the last command
for a co-partitioned implementation. `MATRICES`, `PARTITIONS`, `CORES` and
`SCHEMES` select the configurations in each job.

Optional core-scaling example, with a fixed matrix and partition count:

```sh
for c in 1 2 4 8; do
  MATRICES=D6-6 PARTITIONS=16 CORES=$c sbatch sweep.sbatch
done
```

Each Slurm sweep writes to `results/<jobid>/`. A local sweep writes to
`results/local/`; `OUT_DIR` selects another output folder.

| Output | Contents |
| --- | --- |
| `spark.csv` | One row per run: dimensions, partitions, resource settings, timing and correctness. |
| `shuffle.csv` | Stage counts and shuffle read/write bytes for completed applications. |
| `<matrix>-<scheme>-c<cores>-p<partitions>.json` | Summary and individual iteration times. |
| `<matrix>-<scheme>-c<cores>-p<partitions>.log` | Program output and lineage. |
| `event-logs/` | Spark event logs used by `tools/event_metrics.py`. |

Join the two CSV files on `app_id`. Event metrics have separate `setup`,
`warmup-*` and `measure-*` groups; use `measure-*` for per-SpMV comparisons.
Shuffle read bytes include local and remote reads. `toDebugString` shows the
RDD dependencies; event logs provide the measured stage and byte counts.

### Timing and command-line options

Each timed SpMV builds a fresh result RDD and completes a `collect()` action.
Input loading, SparkContext creation, input setup and verification are outside
that interval. Loading, SparkContext creation and setup have separate timing
fields; `sequential_ms` records the Python correctness baseline.

| Option | Default | Meaning |
| --- | --- | --- |
| `--scheme` | `join` | `join` or `broadcast`; a co-partitioned implementation adds `copartition`. |
| `--partitions N` | 4 | Number of A partitions, also passed to the SpMV functions. |
| `--warmup N` / `--iters N` | 1 / 3 | Warmup and timed iterations; the sweep uses 5 timed iterations. |
| `--csv PATH` | — | Append a result row, adding the header to a new file. |
| `--json PATH` | — | Save the summary and iteration times. |
| `--event-log-dir PATH` | — | Save Spark event logs. |

## Compare with HW1–HW3

[hw4.md](hw4.md) lists the required analysis. Compare with your HW1–HW3 results
on the same matrices. This harness uses Python doubles and a seeded Python
random generator; the earlier C harnesses use float32 and C's random generator.

## Submit

Create `HW4-unityID.zip` containing this folder:

```text
HW4-unityID/
  spmv-spark.py
  report.pdf
  plots/
```

Use the filename `spmv-spark.py` and place it alongside `report.pdf`. Include the
plots in the PDF and as separate files, either in `plots/` or alongside the PDF.
The submission contains your Python script, report and plots.

References: [Spark RDD guide](https://spark.apache.org/docs/3.5.7/rdd-programming-guide.html),
[Spark monitoring](https://spark.apache.org/docs/3.5.7/monitoring.html).
