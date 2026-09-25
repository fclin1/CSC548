#!/usr/bin/env python3
"""
generate_plots.py - Generates all required figures for HW2 report.
Outputs high-resolution PNGs into results/plots/
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def matrix_shortname(path):
    return os.path.basename(path).replace(".mtx", "")

# ============================================================
# Figure 1: Memory Comparison (Task 5: CSR vs ELL Storage)
# ============================================================
def plot_figure_1_memory():
    print("Generating Figure 1: CSR vs ELL Memory Comparison...")
    csv_file = os.path.join(RESULTS_DIR, "scaling_omp_ell.csv")
    if not os.path.exists(csv_file):
        print(f"Skipping Fig 1: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    df = df[df['threads'] == 1].drop_duplicates(subset=['matrix']).copy()
    df['mat_name'] = df['matrix'].apply(matrix_shortname)

    # Compute storage in MB
    # CSR: nnz * 8 + (rows + 1) * 4
    df['csr_mb'] = (df['nnz'] * 8 + (df['rows'] + 1) * 4) / 1e6
    # ELL: ell_slots * 8 (4 byte val + 4 byte col_idx)
    df['ell_mb'] = (df['ell_slots'] * 8) / 1e6
    df['ratio'] = df['ell_mb'] / df['csr_mb']

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.35

    rects1 = ax1.bar(x - width/2, df['csr_mb'], width, label='CSR Storage (MB)', color='#2b5c8f')
    rects2 = ax1.bar(x + width/2, df['ell_mb'], width, label='ELL Storage (MB)', color='#e26d5c')

    ax1.set_ylabel('Memory Footprint (MB)', fontweight='bold')
    ax1.set_title('Task 5: Storage Footprint Comparison (CSR vs Column-Major ELL)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['mat_name'])
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')

    # Add text labels on top of bars showing padding percentage
    for i, row in df.reset_index().iterrows():
        pad = row['pad_pct']
        ratio = row['ratio']
        y_pos = max(row['ell_mb'], row['csr_mb']) * 1.3
        ax1.text(i, y_pos, f"Pad: {pad:.1f}%\n({ratio:.1f}x)", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333333')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig1_csr_vs_ell_memory.png"))
    plt.close()
    print("Saved fig1_csr_vs_ell_memory.png")

# ============================================================
# Figure 2: Thread Strong-Scaling (Task 6: Speedup & Efficiency)
# ============================================================
def plot_figure_2_scaling():
    print("Generating Figure 2: Thread Strong-Scaling...")
    csv_file = os.path.join(RESULTS_DIR, "scaling_omp_csr.csv")
    if not os.path.exists(csv_file):
        print(f"Skipping Fig 2: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    df['mat_name'] = df['matrix'].apply(matrix_shortname)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    threads = sorted(df['threads'].unique())
    # Plot ideal line
    ax1.plot(threads, threads, 'k--', label='Ideal Linear', alpha=0.6)
    ax2.plot(threads, [1.0]*len(threads), 'k--', label='Ideal (1.0)', alpha=0.6)

    colors = plt.cm.tab10(np.linspace(0, 1, len(df['mat_name'].unique())))

    for (mat, group), color in zip(df.groupby('mat_name'), colors):
        group = group.sort_values('threads')
        t1_time = group[group['threads'] == 1]['ms_per_iter'].values[0]
        speedup = t1_time / group['ms_per_iter'].values
        efficiency = speedup / group['threads'].values

        ax1.plot(group['threads'], speedup, marker='o', label=mat, color=color, linewidth=1.8)
        ax2.plot(group['threads'], efficiency, marker='s', label=mat, color=color, linewidth=1.8)

    ax1.set_xlabel('Thread Count (t)', fontweight='bold')
    ax1.set_ylabel('Speedup S(t) = T1 / Tt', fontweight='bold')
    ax1.set_title('Strong Scaling Speedup (CSR)', fontweight='bold')
    ax1.set_xticks(threads)
    ax1.legend()

    ax2.set_xlabel('Thread Count (t)', fontweight='bold')
    ax2.set_ylabel('Parallel Efficiency E(t) = S(t) / t', fontweight='bold')
    ax2.set_title('Parallel Efficiency (CSR)', fontweight='bold')
    ax2.set_xticks(threads)
    ax2.set_ylim(0, 1.2)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig2_thread_scaling.png"))
    plt.close()
    print("Saved fig2_thread_scaling.png")

# ============================================================
# Figure 3: Load Imbalance vs Runtime (Task 3: Load Balancing)
# ============================================================
def plot_figure_3_load_balancing():
    print("Generating Figure 3: Load Imbalance vs Wall-Clock Runtime...")
    csv_file = os.path.join(RESULTS_DIR, "load_balance_csr.csv")
    if not os.path.exists(csv_file):
        print(f"Skipping Fig 3: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    df['mat_name'] = df['matrix'].apply(matrix_shortname)
    df['config'] = df.apply(lambda r: f"{r['partition']}-{r['sched_kind']}" + (f",{r['sched_chunk']}" if r['sched_chunk'] > 0 else ""), axis=1)

    mats = df['mat_name'].unique()
    fig, axes = plt.subplots(len(mats), 1, figsize=(11, 4 * len(mats)), squeeze=False)

    for i, mat in enumerate(mats):
        ax1 = axes[i][0]
        sub = df[df['mat_name'] == mat].copy().reset_index(drop=True)
        x = np.arange(len(sub))
        width = 0.38

        color_time = '#1f77b4'
        color_imb = '#d62728'

        ax1.bar(x - width/2, sub['ms_per_iter'], width, label='Execution Time (ms)', color=color_time)
        ax1.set_ylabel('Execution Time (ms)', color=color_time, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color_time)
        ax1.set_xticks(x)
        ax1.set_xticklabels(sub['config'], rotation=25, ha='right')
        ax1.set_title(f'Load Balancing on High-Variance Matrix: {mat} (8 threads)', fontweight='bold')

        ax2 = ax1.twinx()
        ax2.plot(x + width/2, sub['imbalance'], color=color_imb, marker='D', linewidth=2, markersize=7, label='Imbalance (max/avg)')
        ax2.set_ylabel('Imbalance Ratio', color=color_imb, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_imb)
        ax2.set_ylim(0.9, max(sub['imbalance']) * 1.3)
        ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig3_load_imbalance.png"))
    plt.close()
    print("Saved fig3_load_imbalance.png")

# ============================================================
# Figure 4: SIMD Speedup (Task 4: Scalar vs SIMD)
# ============================================================
def plot_figure_4_simd_speedup():
    print("Generating Figure 4: SIMD Speedup across Formats...")
    csr_scal_file = os.path.join(RESULTS_DIR, "simd_compare_csr_scalar.csv")
    csr_simd_file = os.path.join(RESULTS_DIR, "simd_compare_csr_simd.csv")
    ell_scal_file = os.path.join(RESULTS_DIR, "simd_compare_ell_scalar.csv")
    ell_simd_file = os.path.join(RESULTS_DIR, "simd_compare_ell_simd.csv")

    if not all(os.path.exists(f) for f in [csr_scal_file, csr_simd_file, ell_scal_file, ell_simd_file]):
        print("Skipping Fig 4: Missing SIMD comparison CSV files.")
        return

    df_csr_scal = pd.read_csv(csr_scal_file)
    df_csr_simd = pd.read_csv(csr_simd_file)
    df_ell_scal = pd.read_csv(ell_scal_file)
    df_ell_simd = pd.read_csv(ell_simd_file)

    df_csr_scal['mat_name'] = df_csr_scal['matrix'].apply(matrix_shortname)
    df_csr_simd['mat_name'] = df_csr_simd['matrix'].apply(matrix_shortname)
    df_ell_scal['mat_name'] = df_ell_scal['matrix'].apply(matrix_shortname)
    df_ell_simd['mat_name'] = df_ell_simd['matrix'].apply(matrix_shortname)

    matrices = df_csr_scal['mat_name'].tolist()
    csr_speedups = []
    ell_speedups = []

    for mat in matrices:
        t_csr_s = df_csr_scal[df_csr_scal['mat_name'] == mat]['ms_per_iter'].values[0]
        t_csr_v = df_csr_simd[df_csr_simd['mat_name'] == mat]['ms_per_iter'].values[0]
        t_ell_s = df_ell_scal[df_ell_scal['mat_name'] == mat]['ms_per_iter'].values[0]
        t_ell_v = df_ell_simd[df_ell_simd['mat_name'] == mat]['ms_per_iter'].values[0]

        csr_speedups.append(t_csr_s / t_csr_v)
        ell_speedups.append(t_ell_s / t_ell_v)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(matrices))
    width = 0.35

    rects1 = ax.bar(x - width/2, csr_speedups, width, label='CSR SIMD Speedup (Along-Row)', color='#3470a3')
    rects2 = ax.bar(x + width/2, ell_speedups, width, label='ELL SIMD Speedup (Across-Row)', color='#e76f51')

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Speedup S = T_scalar / T_simd (8 threads)', fontweight='bold')
    ax.set_title('Task 4: SIMD Vectorization Speedup (CSR vs Column-Major ELL)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(matrices)
    ax.legend()

    for rects in [rects1, rects2]:
        for r in rects:
            h = r.get_height()
            ax.annotate(f'{h:.2f}x',
                        xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig4_simd_speedup.png"))
    plt.close()
    print("Saved fig4_simd_speedup.png")

# ============================================================
# Figure 5: Roofline Model (Task 6: Empirical Roofline Analysis)
# ============================================================
def plot_figure_5_roofline():
    print("Generating Figure 5: Roofline Model...")
    combined_file = os.path.join(RESULTS_DIR, "all_benchmarks_combined.csv")
    if not os.path.exists(combined_file):
        print(f"Skipping Fig 5: {combined_file} not found.")
        return

    df = pd.read_csv(combined_file)

    # Machine specs (EPYC 7302P compute node)
    # Peak FP32 GFLOP/s = 16 cores * 2 FMAs * 8 lanes * 3.0 GHz = 768 GFLOP/s
    peak_gflops = 768.0
    # Peak DRAM Bandwidth: 8 channels * 3200 MT/s * 8 B = 204.8 GB/s (practical achievable ~160 GB/s)
    peak_bandwidth_gbs = 160.0

    fig, ax = plt.subplots(figsize=(10, 6))

    ai_range = np.logspace(-2, 2, 500)
    # Roofline boundary: min(Peak_GFLOPs, AI * Peak_BW)
    roofline = np.minimum(peak_gflops, ai_range * peak_bandwidth_gbs)
    ax.plot(ai_range, roofline, 'k-', linewidth=2.5, label='Machine Roofline (EPYC 7302P)')
    ax.text(0.015, peak_bandwidth_gbs * 0.015 * 1.3, f'Memory Bandwidth Ceiling: {peak_bandwidth_gbs:.0f} GB/s',
            rotation=38, fontweight='bold', color='black', fontsize=9)
    ax.axhline(peak_gflops, color='k', linestyle=':', alpha=0.5)
    ax.text(2.0, peak_gflops * 1.08, f'Peak Compute Ceiling: {peak_gflops:.0f} GFLOP/s',
            fontweight='bold', color='black', fontsize=9)

    # SpMV Arithmetic Intensity line (~0.17 FLOP/byte)
    ax.axvline(0.17, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='SpMV AI ~0.17 FLOP/byte')

    # Scatter experimental points
    # Calculate empirical AI = gflops / gbytes
    df['ai'] = df['gflops'] / df['gbytes']

    markers = {'spmv_omp_csr': 'o', 'spmv_omp_ell': '^', 'spmv_simd_csr': 's', 'spmv_simd_ell': 'D'}
    colors = {'spmv_omp_csr': '#1f77b4', 'spmv_omp_ell': '#2ca02c', 'spmv_simd_csr': '#ff7f0e', 'spmv_simd_ell': '#d62728'}

    for kernel, group in df.groupby('kernel'):
        ax.scatter(group['ai'], group['gflops'], label=kernel, marker=markers.get(kernel, 'o'),
                   color=colors.get(kernel, 'gray'), alpha=0.7, s=40)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.01, 20)
    ax.set_ylim(0.1, 1500)
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontweight='bold')
    ax.set_ylabel('Attained Performance (GFLOP/s)', fontweight='bold')
    ax.set_title('Task 6: Empirical Roofline Model for SpMV Kernels', fontweight='bold')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig5_roofline.png"))
    plt.close()
    print("Saved fig5_roofline.png")

if __name__ == '__main__':
    plot_figure_1_memory()
    plot_figure_2_scaling()
    plot_figure_3_load_balancing()
    plot_figure_4_simd_speedup()
    plot_figure_5_roofline()
    print("All plots generated in results/plots/!")
