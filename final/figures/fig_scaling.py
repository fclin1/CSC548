import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.makedirs("figures", exist_ok=True)

# Okabe-Ito color palette
colors = {
    "kmeans_seq": "#000000",
    "cuda_naive": "#E69F00",
    "cuda_shmem": "#56B4E9",
    "cuda_device": "#009E73"
}

# --- Plot 1: speedup_vs_n ---
if os.path.exists("results/suite_b_n_scaling.csv"):
    df_b = pd.read_csv("results/suite_b_n_scaling.csv")
    seq_df = df_b[df_b["impl"] == "kmeans_seq"].set_index("N")
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    n_vals = sorted(df_b["N"].unique())
    
    for impl, label in [("cuda_naive", "Stage 1: Naive (AoS Global)"),
                        ("cuda_shmem", "Stage 2: Padded SoA + ShMem"),
                        ("cuda_device", "Stage 3: Vectorized float4 + On-Device")]:
        sub = df_b[df_b["impl"] == impl].set_index("N")
        speedups = []
        for n in n_vals:
            if n in sub.index and n in seq_df.index:
                t_seq = seq_df.loc[n, "total_compute_ms"]
                t_gpu = sub.loc[n, "total_compute_ms"]
                speedups.append(t_seq / t_gpu)
            else:
                speedups.append(np.nan)
        ax.plot(n_vals, speedups, marker="o", linewidth=2.2, label=label, color=colors[impl])
    
    # 6x multicore i5 CPU theoretical upper bound line
    ax.axhline(y=6.0, color="#D55E00", linestyle="--", linewidth=1.8, label="Ideal 6-Core CPU Bound (6x)")
    
    ax.set_xscale("log")
    ax.set_xlabel("Problem Size N (UCI SUSY points)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Speedup over Serial Baseline ($T_{seq} / T_{GPU}$)", fontsize=11, fontweight="bold")
    ax.set_title("Strong Scaling & Speedup vs. Problem Size N\n(UCI SUSY, D=18, K=32, I=20)", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/speedup_vs_n.png", dpi=300)
    plt.savefig("figures/speedup_vs_n.pdf")
    plt.close()
    print("Generated figures/speedup_vs_n.{png,pdf}")

# --- Plot 2: gpu_saturation ---
if os.path.exists("results/suite_c_gpu_saturation.csv"):
    df_c = pd.read_csv("results/suite_c_gpu_saturation.csv")
    sub_c1 = df_c[(df_c["impl"] == "cuda_device") & (df_c["dataset"].str.contains("susy"))]
    
    if not sub_c1.empty:
        fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
        ax2 = ax1.twinx()
        
        n_pts = sub_c1["N"].values
        gflops = sub_c1["gflops"].values
        # Points per second = (N * iters) / (total_compute_ms / 1000)
        pts_per_sec = (n_pts * sub_c1["iters_run"].values) / (sub_c1["total_compute_ms"].values * 1e-3)
        
        line1 = ax1.plot(n_pts, gflops, marker="s", color="#009E73", linewidth=2.2, label="Compute Rate (GFLOP/s)")
        line2 = ax2.plot(n_pts, pts_per_sec / 1e6, marker="^", color="#CC79A7", linewidth=2.2, linestyle="-.", label="Throughput (Million Pts/s)")
        
        ax1.set_xscale("log")
        ax1.set_xlabel("Problem Size N (Points)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Compute Throughput (GFLOP/s)", fontsize=11, fontweight="bold", color="#009E73")
        ax2.set_ylabel("Point Throughput (M Pts/s)", fontsize=11, fontweight="bold", color="#CC79A7")
        ax1.set_title("GPU Grid Saturation & Scaling Across SM Waves\n(GTX 1080 Ti: 28 SMs, 256 threads/block)", fontsize=12, fontweight="bold")
        ax1.grid(True, which="both", linestyle=":", alpha=0.6)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="lower right", frameon=True, fontsize=10)
        
        plt.tight_layout()
        plt.savefig("figures/gpu_saturation.png", dpi=300)
        plt.savefig("figures/gpu_saturation.pdf")
        plt.close()
        print("Generated figures/gpu_saturation.{png,pdf}")
