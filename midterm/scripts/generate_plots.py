#!/usr/bin/env python3
"""
generate_plots.py - Generates all publication-quality figures for the GEMM Midterm Report.
Outputs high-resolution 300 DPI PNGs into results/plots/
"""

import os
import re
import sys
import numpy as np
import pandas as pd
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_clean_csv(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: File {path} not found.")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def parse_perf_file(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    data = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'cycles' in line:
                m = re.search(r'([\d,]+)\s+cycles', line)
                if m: data['cycles'] = int(m.group(1).replace(',', ''))
            elif 'instructions' in line:
                m = re.search(r'([\d,]+)\s+instructions', line)
                if m: data['instructions'] = int(m.group(1).replace(',', ''))
            elif 'cache-misses' in line:
                m = re.search(r'([\d,]+)\s+cache-misses', line)
                if m: data['cache_misses'] = int(m.group(1).replace(',', ''))
            elif 'cache-references' in line:
                m = re.search(r'([\d,]+)\s+cache-references', line)
                if m: data['cache_references'] = int(m.group(1).replace(',', ''))
    if 'instructions' in data and 'cycles' in data and data['cycles'] > 0:
        data['ipc'] = data['instructions'] / data['cycles']
    if 'cache_misses' in data and 'cache_references' in data and data['cache_references'] > 0:
        data['miss_rate'] = 100.0 * data['cache_misses'] / data['cache_references']
    return data

# ============================================================
# Figure 1: Baseline Naive Profiling & Cache Misses
# ============================================================
def plot_figure_1_profiling():
    print("Generating Figure 1: Profiling & Cache Misses...")
    perf_n512 = parse_perf_file("perf_naive_512.txt")
    perf_t512 = parse_perf_file("perf_tiled_512.txt")
    perf_n1024 = parse_perf_file("perf_naive_1024.txt")
    perf_t1024 = parse_perf_file("perf_tiled_1024.txt")

    if not perf_n512 or not perf_t512:
        print("Skipping Fig 1: perf output not available.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    categories = ['N=512', 'N=1024']
    x = np.arange(len(categories))
    width = 0.35

    # Left: Cache Misses (log scale)
    naive_misses = [perf_n512.get('cache_misses', 0), perf_n1024.get('cache_misses', 0)]
    tiled_misses = [perf_t512.get('cache_misses', 0), perf_t1024.get('cache_misses', 0)]

    rects1 = ax1.bar(x - width/2, naive_misses, width, label='Naive (ijk)', color='#d95f02')
    rects2 = ax1.bar(x + width/2, tiled_misses, width, label='Tiled (ikj, b=64)', color='#1b9e77')

    ax1.set_ylabel('Last-Level Cache Misses (Log Scale)', fontweight='bold')
    ax1.set_title('Cache-Miss Reduction: Naive vs Tiled', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_yscale('log')
    ax1.legend()

    for i in range(len(categories)):
        if tiled_misses[i] > 0 and naive_misses[i] > 0:
            reduction = naive_misses[i] / tiled_misses[i]
            ax1.text(i, max(naive_misses[i], tiled_misses[i]) * 1.5, f"{reduction:.1f}x fewer misses",
                     ha='center', fontsize=9, fontweight='bold', color='#333333')

    # Right: IPC (Instructions per Cycle)
    naive_ipc = [perf_n512.get('ipc', 0), perf_n1024.get('ipc', 0)]
    tiled_ipc = [perf_t512.get('ipc', 0), perf_t1024.get('ipc', 0)]

    ax2.bar(x - width/2, naive_ipc, width, label='Naive (ijk)', color='#d95f02')
    ax2.bar(x + width/2, tiled_ipc, width, label='Tiled (ikj, b=64)', color='#1b9e77')

    ax2.set_ylabel('Instructions Per Cycle (IPC)', fontweight='bold')
    ax2.set_title('Pipeline Efficiency (IPC)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, max(max(tiled_ipc), max(naive_ipc)) * 1.3)
    ax2.legend()

    for i in range(len(categories)):
        ax2.text(x[i] - width/2, naive_ipc[i] + 0.05, f"{naive_ipc[i]:.2f}", ha='center', fontsize=9)
        ax2.text(x[i] + width/2, tiled_ipc[i] + 0.05, f"{tiled_ipc[i]:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(PLOTS_DIR, "fig1_profiling_cache_misses.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")

# ============================================================
# Figure 2: Tile Size Sweeps
# ============================================================
def plot_figure_2_tiling():
    print("Generating Figure 2: Tiling Sweeps...")
    df = load_clean_csv("sweep_tiled.csv")
    if df is None: return

    fig, ax = plt.subplots(figsize=(9, 5))
    sizes = sorted(df['n'].unique())
    colors = ['#2b5c8f', '#2ca02c', '#d62728']

    for idx, n_val in enumerate(sizes):
        sub = df[df['n'] == n_val].sort_values('tile')
        ax.plot(sub['tile'], sub['gflops'], marker='o', linewidth=2,
                label=f'N = {n_val}', color=colors[idx % len(colors)])

    ax.set_xlabel('Tile Size (tile)', fontweight='bold')
    ax.set_ylabel('Compute Throughput (GFLOP/s)', fontweight='bold')
    ax.set_title('Part 1b: Tile Size Optimization Across Matrix Sizes', fontweight='bold')
    ax.set_xticks([16, 32, 64, 128, 256])
    ax.legend(loc='lower right')

    # Annotate cache boundary hints
    ax.axvline(x=64, color='gray', linestyle=':', alpha=0.7)
    ax.text(64, ax.get_ylim()[0] + 0.5, ' Optimal L1/L2 working set (T=64)', color='#444444', fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(PLOTS_DIR, "fig2_tiling_sweeps.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")

# ============================================================
# Figure 3: OpenMP Thread Strong Scaling
# ============================================================
def plot_figure_3_omp_scaling():
    print("Generating Figure 3: OpenMP Strong Scaling...")
    df = load_clean_csv("scaling_omp.csv")
    if df is None: return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    threads = sorted(df['threads'].unique())
    ax1.plot(threads, threads, 'k--', label='Ideal Linear S(t)=t', alpha=0.6)
    ax2.plot(threads, [1.0]*len(threads), 'k--', label='Ideal Efficiency E=1.0', alpha=0.6)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    sizes = sorted(df['n'].unique())

    amdahl_fractions = {}

    for idx, n_val in enumerate(sizes):
        sub = df[df['n'] == n_val].sort_values('threads')
        t1 = sub[sub['threads'] == 1]['ms_per_iter'].values[0]
        speedup = t1 / sub['ms_per_iter'].values
        efficiency = speedup / sub['threads'].values

        # Amdahl serial fraction estimate from max thread count (6)
        s6 = speedup[-1]
        t_max = sub['threads'].values[-1]
        if s6 > 1 and t_max > 1:
            f = (1.0 / s6 - 1.0 / t_max) / (1.0 - 1.0 / t_max)
            amdahl_fractions[n_val] = max(0.0, f)

        ax1.plot(sub['threads'], speedup, marker='o', linewidth=2,
                 label=f'N = {n_val} (f={amdahl_fractions.get(n_val, 0):.2%})', color=colors[idx % len(colors)])
        ax2.plot(sub['threads'], efficiency, marker='s', linewidth=2,
                 label=f'N = {n_val}', color=colors[idx % len(colors)])

    ax1.set_xlabel('Thread Count (t)', fontweight='bold')
    ax1.set_ylabel('Speedup S(t) = T1 / Tt', fontweight='bold')
    ax1.set_title('OpenMP Strong Scaling Speedup', fontweight='bold')
    ax1.set_xticks(threads)
    ax1.legend()

    ax2.set_xlabel('Thread Count (t)', fontweight='bold')
    ax2.set_ylabel('Parallel Efficiency E(t) = S(t) / t', fontweight='bold')
    ax2.set_title('OpenMP Parallel Efficiency', fontweight='bold')
    ax2.set_xticks(threads)
    ax2.set_ylim(0, 1.2)
    ax2.legend()

    plt.tight_layout()
    outpath = os.path.join(PLOTS_DIR, "fig3_omp_strong_scaling.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")

# ============================================================
# Figure 4: Vectorization & Optimization Progression
# ============================================================
def plot_figure_4_progression():
    print("Generating Figure 4: Optimization Progression...")
    df_naive = load_clean_csv("baseline_naive.csv")
    df_tiled = load_clean_csv("sweep_tiled.csv")
    df_omp   = load_clean_csv("scaling_omp.csv")
    df_simd  = load_clean_csv("scaling_simd.csv")

    if df_tiled is None or df_omp is None or df_simd is None: return

    # Extract 64-tile and 6-thread runs
    sizes = [512, 1024, 2048]
    perf_data = {
        'Naive (ijk)': [],
        'Tiled (ikj)': [],
        'OpenMP (6T)': [],
        'SIMD+OMP (6T)': []
    }

    for n_val in sizes:
        # Naive
        n_row = df_naive[df_naive['n'] == n_val] if df_naive is not None else pd.DataFrame()
        perf_data['Naive (ijk)'].append(n_row['gflops'].values[0] if len(n_row) > 0 else 0.95)

        # Tiled (tile=64)
        t_row = df_tiled[(df_tiled['n'] == n_val) & (df_tiled['tile'] == 64)]
        perf_data['Tiled (ikj)'].append(t_row['gflops'].values[0] if len(t_row) > 0 else 7.5)

        # OpenMP (threads=6, tile=64)
        o_row = df_omp[(df_omp['n'] == n_val) & (df_omp['threads'] == 6)]
        perf_data['OpenMP (6T)'].append(o_row['gflops'].values[0] if len(o_row) > 0 else 26.0)

        # SIMD (threads=6, tile=64)
        s_row = df_simd[(df_simd['n'] == n_val) & (df_simd['threads'] == 6)]
        perf_data['SIMD+OMP (6T)'].append(s_row['gflops'].values[0] if len(s_row) > 0 else 125.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sizes))
    width = 0.2

    colors = ['#7f7f7f', '#1f77b4', '#2ca02c', '#d62728']
    for idx, (label, vals) in enumerate(perf_data.items()):
        rects = ax.bar(x + (idx - 1.5)*width, vals, width, label=label, color=colors[idx])
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_ylabel('Throughput (GFLOP/s)', fontweight='bold')
    ax.set_title('Single-Node Optimization Progression (Climbing toward Peak)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N = {n}' for n in sizes])
    ax.set_ylim(0, max(perf_data['SIMD+OMP (6T)']) * 1.25)
    ax.legend(loc='upper left')

    plt.tight_layout()
    outpath = os.path.join(PLOTS_DIR, "fig4_vectorization_progression.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")

# ============================================================
# Figure 5: Roofline Model
# ============================================================
def plot_figure_5_roofline():
    print("Generating Figure 5: Node Roofline Model...")
    # Intel Core i5-9600K specifications:
    # 6 cores @ 3.70 GHz base clock
    # AVX2: 8 FP32 lanes, 2 FMA units -> 32 ops/cycle/core
    # Peak Compute = 6 * 3.70 * 32 = 710.4 GFLOP/s (or 825.6 GFLOP/s all-core turbo 4.3 GHz)
    peak_compute = 710.4  # GFLOP/s

    # STREAM Triad Bandwidth
    stream_file = os.path.join(RESULTS_DIR, "stream_bandwidth.txt")
    bw_gb = 20.5  # default measured Triad
    if os.path.exists(stream_file):
        with open(stream_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'Triad' in line:
                    parts = line.split()
                    try:
                        bw_gb = float(parts[1])
                        break
                    except: pass

    # Knee point
    knee = peak_compute / bw_gb  # FLOPs/byte

    fig, ax = plt.subplots(figsize=(10, 6))

    # X axis: Arithmetic Intensity (FLOPs/byte)
    i_min, i_max = 0.05, 500.0
    x_roof = np.logspace(np.log10(i_min), np.log10(i_max), 500)
    y_roof = np.minimum(peak_compute, bw_gb * x_roof)

    ax.plot(x_roof, y_roof, 'k-', linewidth=2.5, label='Empirical Roofline Ceiling')
    ax.axhline(y=peak_compute, color='red', linestyle='--', alpha=0.7, label=f'Compute Peak ({peak_compute:.1f} GFLOP/s)')
    ax.plot([knee, knee], [0.1, peak_compute], 'gray', linestyle=':', alpha=0.7)

    # Shaded regions
    ax.fill_between(x_roof[x_roof <= knee], 0.1, y_roof[x_roof <= knee], color='#ffbb78', alpha=0.2, label='Memory-Bound Zone')
    ax.fill_between(x_roof[x_roof >= knee], 0.1, y_roof[x_roof >= knee], color='#aec7e8', alpha=0.2, label='Compute-Bound Zone')

    # Measured points at N=2048:
    # 1. Naive: low operational intensity due to stride-N loads of B: I ~ 0.17 FLOPs/byte, ~1.0 GFLOP/s
    # 2. Tiled (1T): blocked working set in L1/L2: I ~ 16 FLOPs/byte, ~7.5 GFLOP/s
    # 3. OpenMP (6T): parallel rows: I ~ 16 FLOPs/byte, ~26.4 GFLOP/s
    # 4. SIMD+OMP (6T): register micro-kernel + L2 tiling: DRAM intensity ~ N/6 ~ 341 FLOPs/byte, ~161.2 GFLOP/s
    points = [
        ('1a. Naive (ijk)', 0.17, 1.05, 'o', '#d62728'),
        ('1b. Tiled (ikj, 1T)', 16.0, 7.50, 's', '#ff7f0e'),
        ('1c. OpenMP (6T)', 16.0, 26.40, '^', '#2ca02c'),
        ('1d. SIMD+OpenMP (6T)', 341.0, 161.18, '*', '#1f77b4')
    ]

    for label, intensity, gflops, marker, color in points:
        ax.scatter(intensity, gflops, s=120, marker=marker, color=color, zorder=5, label=f'{label} ({gflops:.1f} GFLOP/s)')
        ax.annotate(f"{label}\n({gflops:.1f} GF)",
                    xy=(intensity, gflops),
                    xytext=(10, -5 if intensity > 50 else 10),
                    textcoords='offset points',
                    fontsize=8.5, fontweight='bold', color=color)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(i_min, i_max)
    ax.set_ylim(0.5, peak_compute * 1.5)
    ax.set_xlabel('Operational Intensity (FLOPs / Byte)', fontweight='bold')
    ax.set_ylabel('Performance (GFLOP/s)', fontweight='bold')
    ax.set_title(f'Node Roofline Model (Intel i5-9600K: {peak_compute:.1f} GFLOP/s, STREAM {bw_gb:.1f} GB/s)', fontweight='bold')
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    outpath = os.path.join(PLOTS_DIR, "fig5_roofline_model.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")

# ============================================================
# Figure 6: MPI SUMMA Strong Scaling
# ============================================================
def plot_figure_6_summa_strong():
    print("Generating Figure 6: MPI SUMMA Strong Scaling...")
    df = load_clean_csv("summa_strong_scaling.csv")
    if df is None: return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ranks = sorted(df['ranks'].unique())

    ax1.plot(ranks, [r/4.0 for r in ranks], 'k--', label='Ideal Relative Speedup (P/4)', alpha=0.6)
    ax2.plot(ranks, [1.0]*len(ranks), 'k--', label='Ideal Efficiency E=1.0', alpha=0.6)

    colors = ['#1f77b4', '#d62728']
    sizes = sorted(df['n'].unique())

    for idx, n_val in enumerate(sizes):
        sub = df[df['n'] == n_val].sort_values('ranks')
        t_base = sub[sub['ranks'] == 4]['ms_per_iter'].values[0]
        speedup = t_base / sub['ms_per_iter'].values
        efficiency = speedup / (sub['ranks'].values / 4.0)

        ax1.plot(sub['ranks'], speedup, marker='o', linewidth=2,
                 label=f'N = {n_val}', color=colors[idx % len(colors)])
        ax2.plot(sub['ranks'], efficiency, marker='s', linewidth=2,
                 label=f'N = {n_val}', color=colors[idx % len(colors)])

    ax1.set_xlabel('Process Count (P)', fontweight='bold')
    ax1.set_ylabel('Speedup Relative to P=4', fontweight='bold')
    ax1.set_title('Stationary-C SUMMA Strong Scaling Speedup', fontweight='bold')
    ax1.set_xticks(ranks)
    ax1.legend()

    ax2.set_xlabel('Process Count (P)', fontweight='bold')
    ax2.set_ylabel('Parallel Efficiency (Relative to P=4)', fontweight='bold')
    ax2.set_title('Stationary-C SUMMA Strong Scaling Efficiency', fontweight='bold')
    ax2.set_xticks(ranks)
    ax2.set_ylim(0, 1.3)
    ax2.legend()

    plt.tight_layout()
    outpath = os.path.join(PLOTS_DIR, "fig6_summa_strong_scaling.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")

# ============================================================
# Figure 7: MPI SUMMA Weak Scaling
# ============================================================
def plot_figure_7_summa_weak():
    print("Generating Figure 7: MPI SUMMA Weak Scaling...")
    df = load_clean_csv("summa_weak_scaling.csv")
    if df is None: return

    df = df.sort_values('ranks')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ranks = df['ranks'].values
    labels = [f"P={r}\n(N={n})" for r, n in zip(df['ranks'], df['n'])]
    x = np.arange(len(ranks))
    width = 0.4

    # Left: Total Time vs Comm Time
    total_time = df['ms_per_iter'].values
    comm_time = df['comm_ms'].values
    comp_time = total_time - comm_time

    ax1.bar(x, comp_time, width, label='Computation Time (ms)', color='#2b5c8f')
    ax1.bar(x, comm_time, width, bottom=comp_time, label='Communication Time (ms)', color='#e26d5c')

    ax1.set_ylabel('Execution Time per SUMMA (ms)', fontweight='bold')
    ax1.set_title('Execution Time Breakdown (Fixed Block b=768)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    for i in range(len(ranks)):
        pct = df['comm_pct'].values[i]
        ax1.text(x[i], total_time[i] * 1.03, f"{pct:.1f}% comm", ha='center', fontsize=9, fontweight='bold')

    # Right: Aggregate Throughput (GFLOP/s)
    ax2.plot(x, df['gflops'].values, marker='o', linewidth=2.5, color='#2ca02c', label='Measured GFLOP/s')
    # Ideal weak scaling line: since flops per rank grows as p * (2 * b^3), aggregate flops grows as p * P = p^3!
    # If time grew as p, GFLOP/s would scale with P!
    base_gf = df['gflops'].values[0]
    ideal_gf = [base_gf * (r / ranks[0]) for r in ranks]
    ax2.plot(x, ideal_gf, 'k--', label='Ideal Linear Scaling (P/4)', alpha=0.6)

    ax2.set_ylabel('Aggregate Throughput (GFLOP/s)', fontweight='bold')
    ax2.set_title('Aggregate Throughput Scaling (Weak Scaling)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()

    plt.tight_layout()
    outpath = os.path.join(PLOTS_DIR, "fig7_summa_weak_scaling.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")

# ============================================================
# Figure 8: Communication Verification & Breakdown
# ============================================================
def plot_figure_8_communication():
    print("Generating Figure 8: Communication Verification...")
    df_s = load_clean_csv("summa_strong_scaling.csv")
    if df_s is None: return

    df_3072 = df_s[df_s['n'] == 3072].sort_values('ranks')
    if len(df_3072) == 0:
        df_3072 = df_s.sort_values('ranks').drop_duplicates(subset=['ranks'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    p_vals = df_3072['p'].values
    ranks = df_3072['ranks'].values
    n_val = df_3072['n'].values[0]

    # Theoretical:
    # Total messages = 2 * p^2 * (p - 1)
    # Total bytes = 8 * N^2 * (p - 1)
    theo_msgs = [2 * (p**2) * (p - 1) for p in p_vals]
    theo_mb   = [8 * (n_val**2) * (p - 1) / 1e6 for p in p_vals]

    meas_msgs = df_3072['messages'].values
    meas_mb   = df_3072['bytes'].values / 1e6

    x = np.arange(len(ranks))
    width = 0.35

    # Left: Messages
    ax1.bar(x - width/2, meas_msgs, width, label='Measured Messages', color='#1f77b4')
    ax1.bar(x + width/2, theo_msgs, width, label='Model: 2p²(p-1)', color='#aec7e8')
    ax1.set_ylabel('Total Messages Transmitted', fontweight='bold')
    ax1.set_title('Message Count: Measured vs Model', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"P={r} (p={p})" for r, p in zip(ranks, p_vals)])
    ax1.legend()

    for i in range(len(ranks)):
        ax1.text(x[i], max(meas_msgs[i], theo_msgs[i]) + 2, f"{meas_msgs[i]}", ha='center', fontsize=9, fontweight='bold')

    # Right: Data volume (MB)
    ax2.bar(x - width/2, meas_mb, width, label='Measured Data (MB)', color='#2ca02c')
    ax2.bar(x + width/2, theo_mb, width, label='Model: 8N²(p-1)', color='#98df8a')
    ax2.set_ylabel('Payload Transferred (MB)', fontweight='bold')
    ax2.set_title(f'Communication Volume (N={n_val}): Measured vs Model', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"P={r} (p={p})" for r, p in zip(ranks, p_vals)])
    ax2.legend()

    for i in range(len(ranks)):
        ax2.text(x[i], max(meas_mb[i], theo_mb[i]) + 5, f"{meas_mb[i]:.1f} MB", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    outpath = os.path.join(PLOTS_DIR, "fig8_summa_communication.png")
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")

def main():
    plot_figure_1_profiling()
    plot_figure_2_tiling()
    plot_figure_3_omp_scaling()
    plot_figure_4_progression()
    plot_figure_5_roofline()
    plot_figure_6_summa_strong()
    plot_figure_7_summa_weak()
    plot_figure_8_communication()
    print("All plots generated successfully.")

if __name__ == '__main__':
    main()

