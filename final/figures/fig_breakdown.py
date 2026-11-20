import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.makedirs("figures", exist_ok=True)

if os.path.exists("results/suite_b_n_scaling.csv"):
    df_b = pd.read_csv("results/suite_b_n_scaling.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    for ax, n_val, title in [(ax1, 1000000, "N = 1,000,000 Points"),
                             (ax2, 5000000, "N = 5,000,000 Points")]:
        sub_n = df_b[df_b["N"] == n_val].set_index("impl")
        
        stages = ["cuda_naive", "cuda_shmem", "cuda_device"]
        stage_labels = ["Stage 1\nNaive (AoS)", "Stage 2\nPadded SoA", "Stage 3\nOpt Device"]
        
        assign_times = [sub_n.loc[s, "assign_ms"] / sub_n.loc[s, "iters_run"] if s in sub_n.index else 0 for s in stages]
        update_times = [sub_n.loc[s, "update_ms"] / sub_n.loc[s, "iters_run"] if s in sub_n.index else 0 for s in stages]
        xfer_times   = [sub_n.loc[s, "xfer_ms"]   / sub_n.loc[s, "iters_run"] if s in sub_n.index else 0 for s in stages]
        
        x = np.arange(len(stages))
        width = 0.55
        
        p1 = ax.bar(x, assign_times, width, label="Assignment Kernel", color="#E69F00", edgecolor="black", alpha=0.85)
        p2 = ax.bar(x, update_times, width, bottom=assign_times, label="Centroid Update", color="#56B4E9", edgecolor="black", alpha=0.85)
        bottoms = np.array(assign_times) + np.array(update_times)
        p3 = ax.bar(x, xfer_times, width, bottom=bottoms, label="PCIe D2H/H2D Transfers", color="#D55E00", edgecolor="black", alpha=0.85)
        
        # Add value labels on top of bars
        totals = np.array(assign_times) + np.array(update_times) + np.array(xfer_times)
        for i, total in enumerate(totals):
            ax.text(i, total + (totals.max() * 0.02), f"{total:.2f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
            
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels, fontsize=10, fontweight="bold")
        ax.set_ylabel("Execution Time per Iteration (ms)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    plt.suptitle("Per-Iteration Execution Time Breakdown across GPU Optimization Stages\n(UCI SUSY, D=18, K=32, I=20)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/time_breakdown.png", dpi=300)
    plt.savefig("figures/time_breakdown.pdf")
    plt.close()
    print("Generated figures/time_breakdown.{png,pdf}")
