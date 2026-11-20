import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.makedirs("figures", exist_ok=True)

colors = {
    "kmeans_seq": "#000000",
    "cuda_naive": "#E69F00",
    "cuda_shmem": "#56B4E9",
    "cuda_device": "#009E73"
}

# --- Plot 1: scaling_vs_k ---
if os.path.exists("results/suite_d_k_sweep.csv"):
    df_d = pd.read_csv("results/suite_d_k_sweep.csv")
    k_vals = sorted(df_d["K"].unique())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    for impl, label in [("kmeans_seq", "Serial CPU Baseline"),
                        ("cuda_naive", "Stage 1: Naive (AoS)"),
                        ("cuda_shmem", "Stage 2: Padded SoA"),
                        ("cuda_device", "Stage 3: Vectorized float4")]:
        sub = df_d[df_d["impl"] == impl].set_index("K")
        ms_list = [sub.loc[k, "ms_per_iter"] if k in sub.index else np.nan for k in k_vals]
        gflops_list = [sub.loc[k, "gflops"] if k in sub.index else np.nan for k in k_vals]
        
        ax1.plot(k_vals, ms_list, marker="o", linewidth=2.0, label=label, color=colors[impl])
        ax2.plot(k_vals, gflops_list, marker="s", linewidth=2.0, label=label, color=colors[impl])
    
    # Architectural markers
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.axvline(x=33, color="#D55E00", linestyle="--", alpha=0.8, label="Roofline Ridge (K≈33)")
        ax.axvline(x=323, color="#0072B2", linestyle=":", alpha=0.8, label="Path B Switch (K=323)")
        ax.axvline(x=682, color="#CC79A7", linestyle="-.", alpha=0.8, label="48KB Cliff (K=682)")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.set_xlabel("Number of Clusters K", fontsize=11, fontweight="bold")
    
    ax1.set_yscale("log")
    ax1.set_ylabel("Execution Time per Iteration (ms)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Latency per Iteration vs. Cluster Count K", fontsize=11, fontweight="bold")
    
    ax2.set_ylabel("Compute Rate (GFLOP/s)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Effective Compute Rate vs. Cluster Count K", fontsize=11, fontweight="bold")
    
    handles, labels = ax1.get_legend_handles_labels()
    # Unique labels
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc="upper left", frameon=True, fontsize=9)
    
    plt.suptitle("Cluster Count K Scaling & Architectural Cliff Analysis\n(UCI SUSY, N=1M, D=18, I=20)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/scaling_vs_k.png", dpi=300)
    plt.savefig("figures/scaling_vs_k.pdf")
    plt.close()
    print("Generated figures/scaling_vs_k.{png,pdf}")

# --- Plot 2: block_size_tuning ---
if os.path.exists("results/suite_e_block_and_dim.csv"):
    df_e = pd.read_csv("results/suite_e_block_and_dim.csv")
    sub_bs = df_e[df_e["D"] == 18]
    
    if not sub_bs.empty:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        bs_vals = sorted(sub_bs["block_size"].unique())
        
        for impl, label in [("cuda_naive", "Stage 1: Naive (AoS)"),
                            ("cuda_shmem", "Stage 2: Padded SoA"),
                            ("cuda_device", "Stage 3: Vectorized float4")]:
            sub = sub_bs[sub_bs["impl"] == impl].set_index("block_size")
            gflops = [sub.loc[bs, "gflops"] if bs in sub.index else np.nan for bs in bs_vals]
            ax.plot(bs_vals, gflops, marker="o", linewidth=2.2, label=label, color=colors[impl])
            
        ax.set_xlabel("CUDA Thread Block Size (Threads per Block)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Compute Rate (GFLOP/s)", fontsize=11, fontweight="bold")
        ax.set_title("CUDA Thread Block Size Optimization\n(UCI SUSY, N=5M, D=18, K=32, I=20)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        plt.savefig("figures/block_size_tuning.png", dpi=300)
        plt.savefig("figures/block_size_tuning.pdf")
        plt.close()
        print("Generated figures/block_size_tuning.{png,pdf}")

# --- Plot 3: dim_scaling ---
if os.path.exists("results/suite_e_block_and_dim.csv"):
    sub_dim = df_e[df_e["block_size"] == 256]
    
    if not sub_dim.empty:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        d_vals = sorted(sub_dim["D"].unique())
        
        for impl, label in [("kmeans_seq", "Serial CPU Baseline"),
                            ("cuda_naive", "Stage 1: Naive (AoS)"),
                            ("cuda_shmem", "Stage 2: Padded SoA"),
                            ("cuda_device", "Stage 3: Vectorized float4")]:
            sub = sub_dim[sub_dim["impl"] == impl].set_index("D")
            gflops = [sub.loc[d, "gflops"] if d in sub.index else np.nan for d in d_vals]
            ax.plot(d_vals, gflops, marker="s", linewidth=2.0, label=label, color=colors[impl])
            
        ax.set_xlabel("Dimensionality D (Features)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Compute Rate (GFLOP/s)", fontsize=11, fontweight="bold")
        ax.set_title("Performance Scaling across Dimensionality D\n(Synthetic, N=1M, K=32, I=20)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        plt.savefig("figures/dim_scaling.png", dpi=300)
        plt.savefig("figures/dim_scaling.pdf")
        plt.close()
        print("Generated figures/dim_scaling.{png,pdf}")
