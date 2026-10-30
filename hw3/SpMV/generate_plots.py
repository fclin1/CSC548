#!/usr/bin/env python3
"""
generate_plots.py - Generates all publication-quality figures for HW3 report.
Outputs high-resolution PNGs (300 DPI) into results/plots/
"""

import os
import sys
import re
import io
import matplotlib
matplotlib.use('Agg')
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_hardware_info():
    hw_file = os.path.join(RESULTS_DIR, "hardware_info.txt")
    if os.path.exists(hw_file):
        with open(hw_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if "5060" in content:
                return {
                    'name': 'RTX 5060 Ti',
                    'label': 'ARC RTX 5060 Ti (448 GB/s, 28.0 TFLOP/s)',
                    'peak_bw': 448.0,
                    'peak_flops': 28000.0,
                    'knee': 28000.0 / 448.0,
                    'is_arc': True
                }
            elif "V100" in content or "Tesla" in content:
                return {
                    'name': 'Tesla V100',
                    'label': 'Tesla V100 (900 GB/s, 14.1 TFLOP/s)',
                    'peak_bw': 900.0,
                    'peak_flops': 14100.0,
                    'knee': 14100.0 / 900.0,
                    'is_arc': True
                }
    return {
        'name': 'GTX 1080 Ti',
        'label': 'GTX 1080 Ti (484 GB/s, 11.3 TFLOP/s)',
        'peak_bw': 484.0,
        'peak_flops': 11340.0,
        'knee': 11340.0 / 484.0,
        'is_arc': False
    }

def is_v100_primary():
    return get_hardware_info()['is_arc']

def matrix_shortname(path):
    if not isinstance(path, str):
        return str(path)
    return os.path.basename(path).replace(".mtx", "")

def load_clean_csv(filepath):
    """Loads CSV while cleanly stripping out status lines from stdout."""
    if not os.path.exists(filepath):
        return None
    valid_lines = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("Reading sparse matrix"):
                continue
            valid_lines.append(line_str)
    if not valid_lines:
        return None
    return pd.read_csv(io.StringIO("\n".join(valid_lines)))

def load_cpu_baselines():
    """Load CPU baseline ms_per_iter for each matrix."""
    baseline_path = os.path.join(RESULTS_DIR, "cpu_baselines.txt")
    # Fallback to ARC cluster Xeon Silver 4110 measurements if needed
    baselines = {
        'bfly': 0.2709,
        'D6-6': 0.5400,
        'dictionary28': 0.6419,
        'Ga3As3H12': 7.8877,
        'pkustk14': 18.8335,
        'roadNet-CA': 16.6578
    }
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        entries = content.split("file=")
        for entry in entries:
            if not entry.strip():
                continue
            mat_match = re.search(r'matrices/([a-zA-Z0-9_\-]+)\.mtx', entry)
            time_match = re.search(r'benchmarking CSR-SpMV:\s+([0-9\.]+)\s+ms', entry)
            if mat_match and time_match:
                mat = mat_match.group(1)
                t_ms = float(time_match.group(1))
                baselines[mat] = t_ms
    return baselines

# ============================================================
# Figure 1: Memory Footprint & Padding Comparison (CSR vs ELL)
# ============================================================
def plot_figure_1_memory():
    print("Generating Figure 1: CSR vs ELL Memory Comparison...")
    csv_file = os.path.join(RESULTS_DIR, "ell_block_sweep.csv")
    df = load_clean_csv(csv_file)
    if df is None:
        print(f"Skipping Fig 1: {csv_file} not found.")
        return

    df = df[df['block'] == 256].drop_duplicates(subset=['matrix']).copy()
    if df.empty:
        df = df.drop_duplicates(subset=['matrix']).copy()
    df['mat_name'] = df['matrix'].apply(matrix_shortname)

    # Storage in MB
    df['csr_mb'] = (df['nnz'] * 8 + (df['rows'] + 1) * 4) / 1e6
    df['ell_mb'] = (df['ell_slots'] * 8) / 1e6
    df['ratio'] = df['ell_mb'] / df['csr_mb']

    # Sort in consistent presentation order
    order = ['bfly', 'D6-6', 'dictionary28', 'pkustk14', 'roadNet-CA', 'Ga3As3H12']
    df['sort_key'] = df['mat_name'].apply(lambda x: order.index(x) if x in order else 99)
    df = df.sort_values('sort_key').reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    width = 0.35

    rects1 = ax1.bar(x - width/2, df['csr_mb'], width, label='CSR Storage (MB)', color='#2b5c8f')
    rects2 = ax1.bar(x + width/2, df['ell_mb'], width, label='ELL Storage (MB)', color='#e26d5c')

    ax1.set_ylabel('Memory Footprint (MB, Log Scale)', fontweight='bold')
    ax1.set_title('Figure 1: Storage Footprint and Padding Overhead (CSR vs Column-Major ELL)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['mat_name'], rotation=15, ha='right', fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=0.5, top=5000)
    ax1.legend(loc='upper left', frameon=True)

    for i, row in df.iterrows():
        pad = row['pad_pct']
        ratio = row['ratio']
        y_pos = max(row['ell_mb'], row['csr_mb']) * 1.35
        ax1.text(i, y_pos, f"Pad: {pad:.1f}%\n({ratio:.1f}x)", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1A202C')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig1_csr_vs_ell_memory.png"))
    plt.close()
    print("Saved fig1_csr_vs_ell_memory.png")

# ============================================================
# Figure 2: Block Size Sweep (GPU Behavior Analysis)
# ============================================================
def plot_figure_2_block_sweep():
    print("Generating Figure 2: Block Size Sweep (GPU Behavior)...")
    combined_file = os.path.join(RESULTS_DIR, "all_benchmarks_combined.csv")
    df = load_clean_csv(combined_file)
    if df is None:
        print(f"Skipping Fig 2: {combined_file} not found.")
        return

    df['mat_name'] = df['matrix'].apply(matrix_shortname)

    # Focus on key contrasting matrices as required: roadNet-CA vs pkustk14 (and Ga3As3H12)
    contrast_mats = ['roadNet-CA', 'pkustk14', 'Ga3As3H12', 'dictionary28']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    kernel_colors = {'scalar': '#3182ce', 'vector': '#38a169', 'ell': '#dd6b20'}
    kernel_markers = {'scalar': 'o', 'vector': 's', 'ell': '^'}

    for idx, mat in enumerate(contrast_mats):
        ax = axes[idx]
        mat_df = df[df['mat_name'] == mat]

        for k in ['scalar', 'vector', 'ell']:
            k_df = mat_df[mat_df['kernel'] == k].sort_values('block')
            if not k_df.empty:
                ax.plot(k_df['block'], k_df['gbytes'], marker=kernel_markers[k],
                        color=kernel_colors[k], linewidth=2, markersize=7, label=f'{k.capitalize()} Kernel')

        ax.set_title(f'Matrix: {mat}', fontweight='bold')
        ax.set_ylabel('Effective Throughput (GB/s)', fontweight='bold')
        ax.set_xticks([64, 128, 256, 512, 1024])
        ax.grid(True, linestyle='--', alpha=0.6)
        if idx == 0:
            ax.legend(frameon=True, loc='best')

    for ax in axes[2:]:
        ax.set_xlabel('Block Size (Threads per Block)', fontweight='bold')

    plt.suptitle('Figure 2: Effect of Thread Block Size on Memory Throughput across Kernels', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig2_block_size_sweep.png"))
    plt.close()
    print("Saved fig2_block_size_sweep.png")

# ============================================================
# Figure 3: Kernel Performance Comparison (Scalar vs Vector vs ELL)
# ============================================================
def plot_figure_3_kernel_comparison():
    print("Generating Figure 3: Scalar vs Vector vs ELL Comparison...")
    combined_file = os.path.join(RESULTS_DIR, "all_benchmarks_combined.csv")
    df = load_clean_csv(combined_file)
    if df is None:
        print(f"Skipping Fig 3: {combined_file} not found.")
        return

    df['mat_name'] = df['matrix'].apply(matrix_shortname)
    df_256 = df[df['block'] == 256].copy()

    order = ['bfly', 'D6-6', 'dictionary28', 'pkustk14', 'roadNet-CA', 'Ga3As3H12']
    matrices = [m for m in order if m in df_256['mat_name'].values]

    scalar_times, vector_times, ell_times = [], [], []
    scalar_bw, vector_bw, ell_bw = [], [], []

    for m in matrices:
        s_row = df_256[(df_256['mat_name'] == m) & (df_256['kernel'] == 'scalar')]
        v_row = df_256[(df_256['mat_name'] == m) & (df_256['kernel'] == 'vector')]
        e_row = df_256[(df_256['mat_name'] == m) & (df_256['kernel'] == 'ell')]

        scalar_times.append(s_row['ms_kernel'].values[0] if not s_row.empty else 0)
        vector_times.append(v_row['ms_kernel'].values[0] if not v_row.empty else 0)
        ell_times.append(e_row['ms_kernel'].values[0] if not e_row.empty else 0)

        scalar_bw.append(s_row['gbytes'].values[0] if not s_row.empty else 0)
        vector_bw.append(v_row['gbytes'].values[0] if not v_row.empty else 0)
        ell_bw.append(e_row['gbytes'].values[0] if not e_row.empty else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(matrices))
    width = 0.26

    # Panel 1: Execution Time (ms, log scale)
    ax1.bar(x - width, scalar_times, width, label='CSR Scalar (1 th/row)', color='#3182ce')
    ax1.bar(x, vector_times, width, label='CSR Vector (1 warp/row)', color='#38a169')
    ax1.bar(x + width, ell_times, width, label='Column-Major ELL', color='#dd6b20')

    ax1.set_yscale('log')
    ax1.set_ylabel('Execution Time (ms, Log Scale)', fontweight='bold')
    ax1.set_title('(a) Kernel Execution Time per SpMV', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(matrices, rotation=15, ha='right', fontweight='bold')
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Panel 2: Achieved Memory Bandwidth (GB/s)
    rects1 = ax2.bar(x - width, scalar_bw, width, label='CSR Scalar', color='#3182ce')
    rects2 = ax2.bar(x, vector_bw, width, label='CSR Vector', color='#38a169')
    rects3 = ax2.bar(x + width, ell_bw, width, label='Column-Major ELL', color='#dd6b20')

    ax2.set_ylabel('Achieved Bandwidth (GB/s)', fontweight='bold')
    ax2.set_title('(b) Effective Memory Throughput (GB/s)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(matrices, rotation=15, ha='right', fontweight='bold')
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Figure 3: Warp Divergence & Memory Coalescing Impact on Kernel Performance (Block=256)', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig3_kernel_performance_comparison.png"))
    plt.close()
    print("Saved fig3_kernel_performance_comparison.png")

# ============================================================
# Figure 4: Speedup vs CPU & PCIe Transfer Overhead
# ============================================================
def plot_figure_4_speedup_and_transfers():
    print("Generating Figure 4: Speedup and Transfer Overhead...")
    combined_file = os.path.join(RESULTS_DIR, "all_benchmarks_combined.csv")
    df = load_clean_csv(combined_file)
    if df is None:
        print(f"Skipping Fig 4: {combined_file} not found.")
        return

    cpu_baselines = load_cpu_baselines()
    df['mat_name'] = df['matrix'].apply(matrix_shortname)
    df_256 = df[df['block'] == 256].copy()

    # Determine best GPU kernel per matrix
    order = ['bfly', 'D6-6', 'dictionary28', 'pkustk14', 'roadNet-CA', 'Ga3As3H12']
    matrices = [m for m in order if m in df_256['mat_name'].values]

    best_kernel_names = []
    speedup_kernel_only = []
    speedup_with_transfer = []
    transfer_pcts = []

    for m in matrices:
        sub = df_256[df_256['mat_name'] == m].copy()
        # Find fastest kernel by ms_kernel
        best_row = sub.sort_values('ms_kernel').iloc[0]
        t_cpu = cpu_baselines.get(m, 1.0)
        t_k = best_row['ms_kernel']
        t_tot = best_row['ms_with_transfer']
        
        s_k = t_cpu / t_k if t_k > 0 else 0
        s_tot = t_cpu / t_tot if t_tot > 0 else 0
        tx_pct = best_row['transfer_pct']

        best_kernel_names.append(f"{m}\n({best_row['kernel']})")
        speedup_kernel_only.append(s_k)
        speedup_with_transfer.append(s_tot)
        transfer_pcts.append(tx_pct)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(matrices))
    width = 0.35

    # Subplot A: Speedup
    rects1 = ax1.bar(x - width/2, speedup_kernel_only, width, label='Speedup (Kernel Only, Resident)', color='#2b6cb0')
    rects2 = ax1.bar(x + width/2, speedup_with_transfer, width, label='Speedup (With H2D + D2H Transfer)', color='#e26d5c')
    ax1.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax1.set_ylabel('Speedup S = T_cpu / T_gpu', fontweight='bold')
    ax1.set_title('(a) Speedup over Sequential CPU Baseline', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(best_kernel_names, fontweight='bold')
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f'{h:.1f}x', xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f'{h:.1f}x', xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Subplot B: Transfer Share (%)
    rects3 = ax2.bar(x, transfer_pcts, width*1.2, color='#718096')
    ax2.set_ylabel('Transfer Time Overhead (%)', fontweight='bold')
    ax2.set_title('(b) PCIe Transfer Time Share (% of End-to-End Run)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(best_kernel_names, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.6)

    for r in rects3:
        h = r.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1A202C')

    plt.suptitle('Figure 4: GPU Speedup over CPU and PCIe Transfer Overhead (Resident vs Non-Resident)', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig4_gpu_speedup_and_transfers.png"))
    plt.close()
    print("Saved fig4_gpu_speedup_and_transfers.png")

# ============================================================
# Figure 5: GPU Empirical Roofline Model
# ============================================================
def plot_figure_5_roofline():
    print("Generating Figure 5: GPU Roofline Model...")
    combined_file = os.path.join(RESULTS_DIR, "all_benchmarks_combined.csv")
    df = load_clean_csv(combined_file)
    if df is None:
        print(f"Skipping Fig 5: {combined_file} not found.")
        return

    df['mat_name'] = df['matrix'].apply(matrix_shortname)
    df_256 = df[df['block'] == 256].copy()

    hw = get_hardware_info()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ai_range = np.logspace(-2, 3, 500)
    
    # Primary measured roofline
    roof_bw = ai_range * hw['peak_bw']
    roof_curve = np.minimum(roof_bw, hw['peak_flops'])
    ax.plot(ai_range, roof_curve, color='#2B6CB0', linewidth=2.5, label=f"{hw['label']}")

    # Course baseline roofline (Tesla V100)
    peak_bw_v100 = 900.0
    peak_flops_v100 = 14100.0
    roof_bw_v100 = ai_range * peak_bw_v100
    roof_v100 = np.minimum(roof_bw_v100, peak_flops_v100)
    if hw['name'] != 'Tesla V100':
        ax.plot(ai_range, roof_v100, color='#718096', linewidth=2, linestyle='--', label='Tesla V100 Course Baseline (900 GB/s, 14.1 TFLOP/s)')

    active_knee = hw['knee']
    ax.axvline(active_knee, color='#2B6CB0', linestyle=':', alpha=0.6)
    ax.text(active_knee*1.1, 15, f'Knee = {active_knee:.1f} FLOP/B', color='#2B6CB0', fontsize=8.5, fontweight='bold')

    # Plot empirical points
    kernel_styles = {
        'scalar': {'color': '#3182ce', 'marker': 'o', 'label': 'CSR Scalar (1 th/row)'},
        'vector': {'color': '#38a169', 'marker': 's', 'label': 'CSR Vector (1 warp/row)'},
        'ell':    {'color': '#dd6b20', 'marker': '^', 'label': 'Column-Major ELL'}
    }

    # Compute AI = GFLOPs / GBYTEs
    df_256['ai'] = df_256['gflops'] / df_256['gbytes']

    for k, style in kernel_styles.items():
        k_df = df_256[df_256['kernel'] == k]
        ax.scatter(k_df['ai'], k_df['gflops'], color=style['color'], marker=style['marker'],
                   s=75, alpha=0.9, zorder=5, label=style['label'])

    # Annotate key matrices
    for idx, row in df_256.iterrows():
        if row['kernel'] in ['vector', 'ell'] and row['mat_name'] in ['Ga3As3H12', 'pkustk14', 'roadNet-CA', 'bfly']:
            ax.annotate(f"{row['mat_name']}", xy=(row['ai'], row['gflops']),
                        xytext=(4, -4), textcoords="offset points", fontsize=8, color='#2D3748')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.02, 200)
    ax.set_ylim(1, 20000)

    ax.set_xlabel('Operational Intensity (FLOP/byte)', fontweight='bold')
    ax.set_ylabel('Attainable Performance (GFLOP/s)', fontweight='bold')
    ax.set_title('Figure 5: Empirical GPU Roofline Model (SpMV Kernels across Test Suite)', fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # Highlight memory-bound region
    ax.axvspan(0.02, active_knee, color='#edf2f7', alpha=0.3, zorder=0)
    ax.text(0.04, 3000, 'MEMORY-BOUND\nREGION (AI ≈ 0.17)', fontsize=11, fontweight='bold', color='#4a5568')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig5_gpu_roofline.png"))
    plt.close()
    print("Saved fig5_gpu_roofline.png")

def main():
    plot_figure_1_memory()
    plot_figure_2_block_sweep()
    plot_figure_3_kernel_comparison()
    plot_figure_4_speedup_and_transfers()
    plot_figure_5_roofline()
    print("All figures successfully generated in", PLOTS_DIR)

if __name__ == "__main__":
    main()
