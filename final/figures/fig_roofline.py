import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.makedirs("figures", exist_ok=True)

# Hardware limits
GPU_PEAK_GFLOPS = 11340.0   # GTX 1080 Ti Boost FP32 GFLOP/s
GPU_PEAK_BW     = 484.0     # GTX 1080 Ti GDDR5X GB/s
CPU_PEAK_GFLOPS = 128.0     # i5-9600K 1 Core AVX2 FP32 GFLOP/s
CPU_PEAK_BW     = 41.6      # DDR4-2666 2-ch GB/s

GPU_RIDGE_POINT = GPU_PEAK_GFLOPS / GPU_PEAK_BW  # ~23.43 FLOP/byte
CPU_RIDGE_POINT = CPU_PEAK_GFLOPS / CPU_PEAK_BW  # ~3.08 FLOP/byte

colors = {
    "kmeans_seq": "#000000",
    "cuda_naive": "#E69F00",
    "cuda_shmem": "#56B4E9",
    "cuda_device": "#009E73"
}

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

# AI axis range
ai_range = np.logspace(-1, 3, 500)

# 1. GPU Roofline Curve
gpu_roof = np.minimum(GPU_PEAK_GFLOPS, GPU_PEAK_BW * ai_range)
ax.plot(ai_range, gpu_roof, color="#009E73", linewidth=2.5, label=f"GTX 1080 Ti Roofline ({GPU_PEAK_GFLOPS:.0f} GFLOP/s, {GPU_PEAK_BW:.0f} GB/s)")

# 2. CPU Roofline Curve
cpu_roof = np.minimum(CPU_PEAK_GFLOPS, CPU_PEAK_BW * ai_range)
ax.plot(ai_range, cpu_roof, color="#555555", linestyle="--", linewidth=1.8, label=f"Intel i5-9600K 1-Core Roofline ({CPU_PEAK_GFLOPS:.0f} GFLOP/s, {CPU_PEAK_BW:.1f} GB/s)")

# Overlay measured data points from Suite D (K sweep)
if os.path.exists("results/suite_d_k_sweep.csv"):
    df_d = pd.read_csv("results/suite_d_k_sweep.csv")
    D = 18
    # Formula: AI = (3 * K * D) / (4 * (D + 1))
    
    for impl, label, marker in [("kmeans_seq", "Serial Baseline", "x"),
                                ("cuda_naive", "Stage 1: Naive (AoS)", "v"),
                                ("cuda_shmem", "Stage 2: Padded SoA", "^"),
                                ("cuda_device", "Stage 3: Vectorized float4", "o")]:
        sub = df_d[df_d["impl"] == impl]
        if not sub.empty:
            k_arr = sub["K"].values
            ai_arr = (3.0 * k_arr * D) / (4.0 * (D + 1.0))
            gflops_arr = sub["gflops"].values
            ax.scatter(ai_arr, gflops_arr, color=colors[impl], marker=marker, s=65, label=f"Measured: {label}", zorder=5)

# Annotate ridge points
ax.axvline(x=GPU_RIDGE_POINT, color="#009E73", linestyle=":", alpha=0.7)
ax.text(GPU_RIDGE_POINT * 1.05, 30, f"GPU Ridge\n{GPU_RIDGE_POINT:.2f} FLOP/B", color="#009E73", fontsize=9, fontweight="bold")

ax.axvline(x=CPU_RIDGE_POINT, color="#555555", linestyle=":", alpha=0.7)
ax.text(CPU_RIDGE_POINT * 0.45, 1.5, f"CPU Ridge\n{CPU_RIDGE_POINT:.2f} FLOP/B", color="#555555", fontsize=9)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.1, 1000)
ax.set_ylim(0.5, 20000)

ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=11, fontweight="bold")
ax.set_ylabel("Attained Performance (GFLOP/s)", fontsize=11, fontweight="bold")
ax.set_title("Empirical Roofline Model: Intel Core i5-9600K vs. NVIDIA GTX 1080 Ti\n(K-Means Clustering on UCI SUSY across K ∈ [8, 1024])", fontsize=12, fontweight="bold")
ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend(loc="upper left", frameon=True, fontsize=9)

plt.tight_layout()
plt.savefig("figures/roofline.png", dpi=300)
plt.savefig("figures/roofline.pdf")
plt.close()
print("Generated figures/roofline.{png,pdf}")
