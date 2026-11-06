#!/usr/bin/env python3
"""Summarize the Spark event logs from spmv-spark.py --event-log-dir by job group.

Counts executed stages (StageSubmitted), not skipped ancestors listed in JobStart.
Logs of unfinished runs (.inprogress) are skipped.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


FIELDS = (
    "app_id", "job_group", "stages", "shuffle_read_bytes", "shuffle_write_bytes",
    "executor_run_ms", "executor_deserialize_ms", "result_serialize_ms",
)


def summarize(path):
    app_id = None
    groups = {}
    stage_groups = {}

    def group_record(name):
        if name not in groups:
            groups[name] = {key: 0 for key in FIELDS[3:]}
            groups[name].update(job_group=name, stages=set())
        return groups[name]

    with Path(path).open() as stream:
        for line in stream:
            event = json.loads(line)
            kind = event["Event"]
            if kind == "SparkListenerApplicationStart":
                app_id = event["App ID"]
            elif kind == "SparkListenerJobStart":
                name = event["Properties"].get("spark.jobGroup.id", "ungrouped")
                group_record(name)
                for stage in event["Stage IDs"]:
                    stage_groups[stage] = name
            elif kind == "SparkListenerStageSubmitted":
                stage = event["Stage Info"]["Stage ID"]
                group_record(stage_groups[stage])["stages"].add(stage)
            elif kind == "SparkListenerTaskEnd":
                record = group_record(stage_groups[event["Stage ID"]])
                metrics = event.get("Task Metrics") or {}
                read = metrics.get("Shuffle Read Metrics", {})
                write = metrics.get("Shuffle Write Metrics", {})
                record["shuffle_read_bytes"] += read.get("Remote Bytes Read", 0) + read.get("Local Bytes Read", 0)
                record["shuffle_write_bytes"] += write.get("Shuffle Bytes Written", 0)
                record["executor_run_ms"] += metrics.get("Executor Run Time", 0)
                record["executor_deserialize_ms"] += metrics.get("Executor Deserialize Time", 0)
                record["result_serialize_ms"] += metrics.get("Result Serialization Time", 0)

    for record in groups.values():
        record["app_id"] = app_id
        record["stages"] = len(record["stages"])
    return list(groups.values())


def log_files(log_dir):
    return [path for path in sorted(Path(log_dir).iterdir()) if not path.name.endswith(".inprogress")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", help="directory given to --event-log-dir")
    args = parser.parse_args()
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for path in log_files(args.log_dir):
        writer.writerows(summarize(path))


if __name__ == "__main__":
    main()
