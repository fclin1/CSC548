#!/usr/bin/env python3
"""Generate publication-quality figures for HW4 Spark SpMV report.

Plots:
  Figure 1: Runtime vs. Partitions (Join vs. Broadcast across 6 matrices)
  Figure 2: Shuffle Communication Volume (Read & Write Bytes vs. Partitions)
  Figure 3: Parallel Scaling & Efficiency (Speedup & Efficiency vs. Partitions)
  Figure 4: Cross-Model Paradigm Comparison (MPI vs. OpenMP vs. CUDA vs. Spark)
"""

import argparse
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "lines.linewidth": 1.75,
    "lines.markersize": 6,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

MATRICES = ["D6-6", "dictionary28", "bfly", "roadNet-CA", "Ga3As3H12", "pkustk14"]
SCHEMES = ["join", "broadcast"]
COLORS = {
    "join": "#d95f02",
    "broadcast": "#1b9e77",
    "mpi": "#386cb0",
    "omp": "#7570b3",
    "cuda": "#e7298a",
}


def clean_matrix_name(path_str):
    """Normalize 'matrices/D6-6.mtx' or 'D6-6.mtx' to 'D6-6'."""
    base = Path(path_str).stem
    return base


def load_spark_data(results_dir):
    spark_csv = Path(results_dir) / "spark.csv"
    if not spark_csv.exists():
        raise FileNotFoundError(f"Missing {spark_csv}")
    df = pd.read_csv(spark_csv)
    df["clean_matrix"] = df["matrix"].apply(clean_matrix_name)
    df["partitions"] = df["partitions"].astype(int)
    df["mean_ms"] = df["mean_ms"].astype(float)
    return df


def load_shuffle_data(results_dir, spark_df):
    shuffle_csv = Path(results_dir) / "shuffle.csv"
    if not shuffle_csv.exists():
        print(f"Warning: {shuffle_csv} not found, shuffle plots may be limited.")
        return None
    shuf = pd.read_csv(shuffle_csv)
    # Filter to timed measures: 'measure-*'
    measure_shuf = shuf[shuf["job_group"].str.startswith("measure-")].copy()
    # Group by app_id to get mean shuffle metrics per SpMV iteration
    agg = measure_shuf.groupby("app_id").agg({
        "stages": "first",
        "shuffle_read_bytes": "mean",
        "shuffle_write_bytes": "mean",
        "executor_run_ms": "mean",
    }).reset_index()
    merged = pd.merge(spark_df, agg, on="app_id", how="left")
    return merged


def plot_runtime_vs_partitions(df, out_dir):
    """Figure 1: Runtime vs. Partitions (Join vs. Broadcast across all 6 matrices)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    axes = axes.flatten()

    for idx, mat in enumerate(MATRICES):
        ax = axes[idx]
        mat_df = df[df["clean_matrix"] == mat]
        if mat_df.empty:
            ax.set_title(f"{mat} (No Data)")
            continue

        for scheme in SCHEMES:
            s_df = mat_df[mat_df["scheme"] == scheme].sort_values("partitions")
            if not s_df.empty:
                marker = "o" if scheme == "broadcast" else "s"
                ax.plot(
                    s_df["partitions"], s_df["mean_ms"],
                    marker=marker, label=f"Scheme: {scheme}",
                    color=COLORS[scheme],
                )

        ax.set_title(mat, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True)
        if idx % 3 == 0:
            ax.set_ylabel("Execution Time (ms, log scale)")
        if idx >= 3:
            ax.set_xlabel("Number of Partitions ($p$)")
        if idx == 0:
            ax.legend(loc="best", frameon=True)

    plt.suptitle("Figure 1: SpMV Execution Time vs. Partition Count (Join vs. Broadcast)", y=0.98, fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = Path(out_dir) / "fig1_runtime_vs_partitions.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_shuffle_volume(df_shuf, out_dir):
    """Figure 2: Shuffle Communication Volume (Read & Write Bytes vs. Partitions)."""
    if df_shuf is None or "shuffle_read_bytes" not in df_shuf.columns:
        print("Skipping Figure 2 (no shuffle data).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Average shuffle read & write across matrices by scheme and partition
    grouped = df_shuf.groupby(["scheme", "partitions"]).agg({
        "shuffle_read_bytes": "mean",
        "shuffle_write_bytes": "mean",
    }).reset_index()

    # Read bytes plot
    ax = axes[0]
    for scheme in SCHEMES:
        s_df = grouped[grouped["scheme"] == scheme].sort_values("partitions")
        if not s_df.empty:
            read_mb = s_df["shuffle_read_bytes"] / (1024 * 1024)
            marker = "o" if scheme == "broadcast" else "s"
            ax.plot(
                s_df["partitions"], read_mb,
                marker=marker, label=f"Scheme: {scheme}",
                color=COLORS[scheme],
            )
    ax.set_title("Mean Shuffle Read Bytes per SpMV", fontweight="bold")
    ax.set_xlabel("Partitions ($p$)")
    ax.set_ylabel("Shuffle Read Volume (MB)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True)
    ax.legend(loc="best", frameon=True)

    # Write bytes plot
    ax = axes[1]
    for scheme in SCHEMES:
        s_df = grouped[grouped["scheme"] == scheme].sort_values("partitions")
        if not s_df.empty:
            write_mb = s_df["shuffle_write_bytes"] / (1024 * 1024)
            marker = "o" if scheme == "broadcast" else "s"
            ax.plot(
                s_df["partitions"], write_mb,
                marker=marker, label=f"Scheme: {scheme}",
                color=COLORS[scheme],
            )
    ax.set_title("Mean Shuffle Write Bytes per SpMV", fontweight="bold")
    ax.set_xlabel("Partitions ($p$)")
    ax.set_ylabel("Shuffle Write Volume (MB)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True)
    ax.legend(loc="best", frameon=True)

    plt.suptitle("Figure 2: Shuffle Communication Traffic vs. Partition Count", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = Path(out_dir) / "fig2_shuffle_volume.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_speedup_and_efficiency(df, out_dir):
    """Figure 3: Strong Scaling Speedup and Parallel Efficiency."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Compute Speedup relative to p=1 for each matrix and scheme
    speedup_records = []
    for mat in MATRICES:
        for scheme in SCHEMES:
            sub = df[(df["clean_matrix"] == mat) & (df["scheme"] == scheme)].sort_values("partitions")
            if sub.empty or (sub["partitions"] == 1).sum() == 0:
                continue
            t1 = sub[sub["partitions"] == 1]["mean_ms"].values[0]
            for _, row in sub.iterrows():
                p = int(row["partitions"])
                speedup = t1 / row["mean_ms"]
                eff = speedup / p
                speedup_records.append({
                    "matrix": mat, "scheme": scheme, "partitions": p,
                    "speedup": speedup, "efficiency": eff,
                })

    if not speedup_records:
        print("Skipping Figure 3 (no p=1 baseline for speedup).")
        return

    sdf = pd.DataFrame(speedup_records)

    # Plot 1: Speedup vs Partitions
    ax = axes[0]
    partitions_range = np.array([1, 2, 4, 8, 16, 32])
    ax.plot(partitions_range, partitions_range, "k--", label="Ideal Linear", alpha=0.7)

    for scheme in SCHEMES:
        sc_df = sdf[sdf["scheme"] == scheme].groupby("partitions")["speedup"].mean().reset_index()
        marker = "o" if scheme == "broadcast" else "s"
        ax.plot(
            sc_df["partitions"], sc_df["speedup"],
            marker=marker, label=f"Mean Speedup ({scheme})",
            color=COLORS[scheme],
        )

    ax.set_title("Parallel Speedup $S(p) = T(1) / T(p)$", fontweight="bold")
    ax.set_xlabel("Number of Partitions ($p$)")
    ax.set_ylabel("Speedup Factor")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True)
    ax.legend(loc="best", frameon=True)

    # Plot 2: Parallel Efficiency vs Partitions
    ax = axes[1]
    ax.axhline(1.0, color="k", linestyle="--", label="Ideal (100%)", alpha=0.7)

    for scheme in SCHEMES:
        sc_df = sdf[sdf["scheme"] == scheme].groupby("partitions")["efficiency"].mean().reset_index()
        marker = "o" if scheme == "broadcast" else "s"
        ax.plot(
            sc_df["partitions"], sc_df["efficiency"] * 100,
            marker=marker, label=f"Mean Efficiency ({scheme})",
            color=COLORS[scheme],
        )

    ax.set_title("Parallel Efficiency $E(p) = S(p) / p$", fontweight="bold")
    ax.set_xlabel("Number of Partitions ($p$)")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 110)
    ax.grid(True)
    ax.legend(loc="best", frameon=True)

    plt.suptitle("Figure 3: Spark Strong Scaling Speedup and Parallel Efficiency", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = Path(out_dir) / "fig3_scaling_and_efficiency.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def load_historical_baselines():
    """Load optimal execution times from HW1 (MPI), HW2 (OpenMP), and HW3 (CUDA)."""
    hw1_data = {
        "D6-6": 0.124, "dictionary28": 0.134, "bfly": 0.080,
        "Ga3As3H12": 1.114, "pkustk14": 2.452, "roadNet-CA": 3.986,
    }
    hw2_data = {
        "D6-6": 0.130, "dictionary28": 0.112, "bfly": 0.050,
        "Ga3As3H12": 1.906, "pkustk14": 3.964, "roadNet-CA": 2.734,
    }
    hw3_data = {
        "D6-6": 0.0078, "dictionary28": 0.0092, "bfly": 0.0066,
        "Ga3As3H12": 0.1236, "pkustk14": 0.2958, "roadNet-CA": 0.1618,
    }
    return hw1_data, hw2_data, hw3_data


def plot_cross_model_comparison(spark_df, out_dir):
    """Figure 4: Cross-Model Paradigm Comparison (HW1 MPI, HW2 OpenMP, HW3 CUDA, HW4 Spark)."""
    hw1, hw2, hw3 = load_historical_baselines()

    # Optimal Spark runtimes per matrix (at optimal partition count)
    spark_join = {}
    spark_bc = {}
    for mat in MATRICES:
        sub = spark_df[spark_df["clean_matrix"] == mat]
        if not sub.empty:
            j_sub = sub[sub["scheme"] == "join"]
            b_sub = sub[sub["scheme"] == "broadcast"]
            if not j_sub.empty:
                spark_join[mat] = j_sub["mean_ms"].min()
            if not b_sub.empty:
                spark_bc[mat] = b_sub["mean_ms"].min()

    x = np.arange(len(MATRICES))
    width = 0.16

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x - 2 * width, [hw3.get(m, 0) for m in MATRICES], width, label="HW3: CUDA (Tesla V100 GPU)", color=COLORS["cuda"])
    ax.bar(x - width, [hw2.get(m, 0) for m in MATRICES], width, label="HW2: OpenMP (8 Threads)", color=COLORS["omp"])
    ax.bar(x, [hw1.get(m, 0) for m in MATRICES], width, label="HW1: MPI (8 Ranks)", color=COLORS["mpi"])
    ax.bar(x + width, [spark_bc.get(m, np.nan) for m in MATRICES], width, label="HW4: Spark Broadcast", color=COLORS["broadcast"])
    ax.bar(x + 2 * width, [spark_join.get(m, np.nan) for m in MATRICES], width, label="HW4: Spark Join", color=COLORS["join"])

    ax.set_ylabel("Execution Time per SpMV (ms, log scale)")
    ax.set_title("Figure 4: Cross-Paradigm Performance Comparison (HW1 MPI vs. HW2 OpenMP vs. HW3 CUDA vs. HW4 Spark)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(MATRICES, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y")
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    out_path = Path(out_dir) / "fig4_cross_model_comparison.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_core_scaling(results_dir, out_dir):
    """Figure 5: Executor Core Strong-Scaling Speedup and Efficiency (cores 1 to 16)."""
    cores_csv = Path(results_dir) / "spark_cores.csv"
    if not cores_csv.exists():
        print("Skipping Figure 5 (no spark_cores.csv).")
        return

    df = pd.read_csv(cores_csv)
    df["clean_matrix"] = df["matrix"].apply(clean_matrix_name)
    df["cores"] = df["master"].str.extract(r"local\[(\d+)\]").astype(int)
    df["mean_ms"] = df["mean_ms"].astype(float)

    # Compute speedup relative to cores=1
    records = []
    for mat in df["clean_matrix"].unique():
        for scheme in df["scheme"].unique():
            sub = df[(df["clean_matrix"] == mat) & (df["scheme"] == scheme)].sort_values("cores")
            if sub.empty or (sub["cores"] == 1).sum() == 0:
                continue
            t1 = sub[sub["cores"] == 1]["mean_ms"].values[0]
            for _, row in sub.iterrows():
                c = int(row["cores"])
                sp = t1 / row["mean_ms"]
                eff = sp / c
                records.append({
                    "matrix": mat, "scheme": scheme, "cores": c,
                    "speedup": sp, "efficiency": eff, "mean_ms": row["mean_ms"]
                })

    if not records:
        return

    sdf = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Speedup vs Cores
    ax = axes[0]
    cores_range = np.array([1, 2, 4, 8, 16])
    ax.plot(cores_range, cores_range, "k--", label="Ideal Linear", alpha=0.7)

    for mat in sdf["matrix"].unique():
        for scheme in ["broadcast", "join"]:
            m_df = sdf[(sdf["matrix"] == mat) & (sdf["scheme"] == scheme)].sort_values("cores")
            if not m_df.empty:
                marker = "o" if scheme == "broadcast" else "s"
                linestyle = "-" if scheme == "broadcast" else ":"
                ax.plot(
                    m_df["cores"], m_df["speedup"],
                    marker=marker, linestyle=linestyle,
                    label=f"{mat} ({scheme})",
                )

    ax.set_title("Core Speedup $S(c) = T(1) / T(c)$ at $p=16$", fontweight="bold")
    ax.set_xlabel("Executor Cores ($c$)")
    ax.set_ylabel("Speedup Factor")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    # Panel 2: Parallel Efficiency vs Cores
    ax = axes[1]
    ax.axhline(100.0, color="k", linestyle="--", label="Ideal (100%)", alpha=0.7)

    for mat in sdf["matrix"].unique():
        for scheme in ["broadcast", "join"]:
            m_df = sdf[(sdf["matrix"] == mat) & (sdf["scheme"] == scheme)].sort_values("cores")
            if not m_df.empty:
                marker = "o" if scheme == "broadcast" else "s"
                linestyle = "-" if scheme == "broadcast" else ":"
                ax.plot(
                    m_df["cores"], m_df["efficiency"] * 100,
                    marker=marker, linestyle=linestyle,
                    label=f"{mat} ({scheme})",
                )

    ax.set_title("Core Parallel Efficiency $E(c) = S(c) / c$", fontweight="bold")
    ax.set_xlabel("Executor Cores ($c$)")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_ylim(0, 115)
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=8, frameon=True)

    plt.suptitle("Figure 5: Executor Core Strong-Scaling Speedup and Efficiency ($p=16$ Partitions)", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = Path(out_dir) / "fig5_core_scaling_and_efficiency.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", help="directory with spark.csv and shuffle.csv")
    parser.add_argument("--out-dir", default="results/plots", help="output directory for plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spark_df = load_spark_data(results_dir)
    shuf_df = load_shuffle_data(results_dir, spark_df)

    plot_runtime_vs_partitions(spark_df, out_dir)
    plot_shuffle_volume(shuf_df, out_dir)
    plot_speedup_and_efficiency(spark_df, out_dir)
    plot_cross_model_comparison(spark_df, out_dir)
    plot_core_scaling(results_dir, out_dir)
    print("All plots generated successfully.")


if __name__ == "__main__":
    main()
