#!/usr/bin/env python3
"""HW4 starter: implement the three TODO functions. Submit this file with your plots and report.pdf."""

import argparse
import csv
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path


# ---- YOUR CODE -------------------------------------------------------------
# Return lazy RDDs and compute products on workers.
# The harness caches inputs and collects each result.


def make_matrix_rdd(sc, entries, partitions):
    """Return an RDD of (row, col, val) with exactly partitions partitions."""
    # TODO (Code 2): distribute A using the requested partition count.
    raise NotImplementedError("implement make_matrix_rdd (Code 2)")


def spmv_join(a_rdd, x_rdd, partitions):
    """Return a lazy RDD of unique (row, y[row]) pairs.

    a_rdd contains (row, col, val); x_rdd contains (col, x[col]).
    Keep the row index when re-keying A by column, join with x, then
    aggregate contributions by row.
    Missing/empty rows may be omitted; the driver fills them with zero.
    """
    # TODO (Code 1): plain column join followed by row reduction.
    raise NotImplementedError("implement spmv_join (Code 1)")


def spmv_broadcast(a_rdd, x_broadcast, partitions):
    """Same output contract as spmv_join; x_broadcast.value is a dense list.

    Read x_broadcast.value inside the worker function, look up x[col],
    and aggregate products by row.
    """
    # TODO (Code 2): broadcast lookup followed
    # by row reduction. Co-partitioning may replace this path: see README.md.
    raise NotImplementedError("implement spmv_broadcast (Code 1/2)")


# ---- PROVIDED: reader, command line, broadcast creation, lineage printing, ----
# timing, checking and output. README.md describes the harness changes for
# the co-partitioning option.


SEED = 13
CSV_FIELDS = (
    "matrix", "rows", "cols", "nnz", "scheme", "partitions", "a_partitions",
    "master", "iters", "warmup", "load_ms", "sequential_ms", "spark_start_ms",
    "setup_ms", "mean_ms", "median_ms", "min_ms", "check", "rel_l2", "app_id",
    "spark_version", "python_version",
)


def read_matrix_market(path):
    """Read zero-based COO entries on the driver; expand symmetry, keep duplicates."""
    with open(path) as stream:
        _, _, _, field, symmetry = stream.readline().lower().split()
        lines = (line.split() for line in stream if line.strip() and not line.startswith("%"))
        rows, cols, _ = map(int, next(lines))
        entries = []
        for fields in lines:
            row, col = int(fields[0]) - 1, int(fields[1]) - 1
            value = 1.0 if field == "pattern" else float(fields[2])
            entries.append((row, col, value))
            if row != col and symmetry != "general":
                entries.append((col, row, -value if symmetry == "skew-symmetric" else value))
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    return rows, cols, entries


def fill_problem(entries, cols):
    """Replace matrix values and generate x from a seeded Python random stream."""
    rng = random.Random(SEED)
    for index, (row, col, _) in enumerate(entries):
        entries[index] = (row, col, 1.0 - 2.0 * rng.random())
    return [rng.random() for _ in range(cols)]


def sequential_spmv(rows, entries, x):
    y = [0.0] * rows
    for row, col, value in entries:
        y[row] += value * x[col]
    return y


def dense_result(pairs, rows):
    y = [0.0] * rows
    for row, value in pairs:
        y[row] = value
    return y


def relative_l2(actual, expected):
    difference = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, expected)))
    return difference / max(math.sqrt(sum(b ** 2 for b in expected)), 1e-30)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def append_csv(path, result):
    is_new = not Path(path).exists()
    with open(path, "a", newline="") as stream:
        writer = csv.writer(stream)
        if is_new:
            writer.writerow(CSV_FIELDS)
        writer.writerow([result[name] for name in CSV_FIELDS])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", help=".mtx file")
    parser.add_argument("--scheme", choices=("join", "broadcast"), default="join")
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--csv", metavar="PATH", help="append one result row")
    parser.add_argument("--json", metavar="PATH", help="summary plus iteration times")
    parser.add_argument("--event-log-dir", metavar="PATH", help="save Spark event logs")
    parser.add_argument("--output-y", metavar="PATH", help="dense result as JSON, used for grading")
    args = parser.parse_args()

    started = time.perf_counter()
    rows, cols, entries = read_matrix_market(args.matrix)
    x = fill_problem(entries, cols)
    load_ms = 1000 * (time.perf_counter() - started)
    started = time.perf_counter()
    expected = sequential_spmv(rows, entries, x)
    sequential_ms = 1000 * (time.perf_counter() - started)

    # Imported here so this file's helpers load without Spark.
    from pyspark import SparkConf, SparkContext, StorageLevel

    conf = SparkConf().setAppName("HW4-SpMV-" + args.scheme)
    if args.event_log_dir:
        log_dir = Path(args.event_log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        conf.set("spark.eventLog.enabled", "true")
        conf.set("spark.eventLog.dir", log_dir.as_uri())
    started = time.perf_counter()
    sc = SparkContext(conf=conf)
    spark_start_ms = 1000 * (time.perf_counter() - started)
    sc.setLogLevel("WARN")
    try:
        started = time.perf_counter()
        sc.setJobGroup("setup", "cache raw A and prepare x")
        a_rdd = make_matrix_rdd(sc, entries, args.partitions)
        a_rdd.persist(StorageLevel.MEMORY_AND_DISK)
        a_rdd.count()
        if args.scheme == "join":
            x_rdd = sc.parallelize(list(enumerate(x)), args.partitions)
            x_rdd.persist(StorageLevel.MEMORY_AND_DISK)
            x_rdd.count()
        else:
            x_broadcast = sc.broadcast(x)
        setup_ms = 1000 * (time.perf_counter() - started)
        nnz = len(entries)
        del entries  # Python tuples can be substantially larger than the .mtx file.

        def spmv():
            if args.scheme == "join":
                return spmv_join(a_rdd, x_rdd, args.partitions)
            return spmv_broadcast(a_rdd, x_broadcast, args.partitions)

        print("LINEAGE (%s):\n%s" % (args.scheme, spmv().toDebugString().decode()), file=sys.stderr)

        elapsed = []
        errors = []
        for iteration in range(-args.warmup, args.iters):
            group = "warmup-%d" % (iteration + args.warmup) if iteration < 0 else "measure-%d" % iteration
            sc.setJobGroup(group, "%s SpMV, partitions=%d" % (args.scheme, args.partitions))
            # A fresh DAG executes each SpMV's shuffle work within its timing.
            started = time.perf_counter()
            pairs = spmv().collect()
            milliseconds = 1000 * (time.perf_counter() - started)
            y = dense_result(pairs, rows)
            errors.append(relative_l2(y, expected))
            if iteration >= 0:
                elapsed.append(milliseconds)

        result = dict(
            matrix=args.matrix, rows=rows, cols=cols, nnz=nnz,
            scheme=args.scheme, partitions=args.partitions,
            a_partitions=a_rdd.getNumPartitions(), master=sc.master,
            iters=args.iters, warmup=args.warmup,
            load_ms=load_ms, sequential_ms=sequential_ms,
            spark_start_ms=spark_start_ms, setup_ms=setup_ms,
            mean_ms=statistics.mean(elapsed), median_ms=statistics.median(elapsed),
            min_ms=min(elapsed), check="PASS" if all(error <= 1e-8 for error in errors) else "FAIL",
            rel_l2=max(errors), app_id=sc.applicationId, spark_version=sc.version,
            python_version=platform.python_version(),
        )
        if args.json:
            write_json(args.json, dict(result, iteration_ms=elapsed))
        if args.output_y:
            write_json(args.output_y, y)
        if args.csv:
            append_csv(args.csv, result)
        print(json.dumps(dict(result, iteration_ms=elapsed), indent=2))
        return 0 if result["check"] == "PASS" else 1
    finally:
        sc.stop()  # Flushes the event log.


if __name__ == "__main__":
    sys.exit(main())
