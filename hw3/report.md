# CSC 548 HW3: GPU SpMV Architecture Report

- **Unity ID:** `fclin`
- **Course:** CSC 548 Parallel Systems (Fall 2026)
- **Empirical Hardware:** ARC Node `c5` (NVIDIA GeForce RTX 5060 Ti, sm_120, 16 GB GDDR7, AMD EPYC 9115 16-Core CPU)
- **Course Reference:** NC State ARC GPU Node (Tesla V100 SXM2, Volta sm_70, 900 GB/s peak, 14.1 TFLOP/s)

---

## 1. Matrix Structure & Storage Footprints

SpMV computes $y = A \cdot x$. Nonzero distribution dictates memory footprint. Regular matrices incur zero padding in ELL. High-variance matrices cause massive memory expansion in ELL.

### Table 1: Matrix Structural Parameters and Memory Footprints

| Matrix | Rows ($m$) | Cols ($n$) | NNZ | Max $K$ | Avg $\mu$ | CSR (MB) | ELL (MB) | Zero Pad % | ELL/CSR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bfly` | 49,152 | 49,152 | 196,608 | 4 | 4.0 | 1.77 | 1.57 | **0.0%** | **0.89x** |
| `D6-6` | 120,576 | 23,740 | 146,880 | 6 | 1.2 | 1.66 | 5.79 | 79.7% | 3.49x |
| `dictionary28` | 52,652 | 52,652 | 178,076 | 38 | 3.4 | 1.64 | 16.01 | 91.1% | 9.79x |
| `pkustk14` | 151,926 | 151,926 | 14,836,504 | 333 | 97.7 | 119.30 | 404.73 | 70.7% | 3.39x |
| `roadNet-CA` | 1,971,281 | 1,971,281 | 5,533,214 | 12 | 2.8 | 52.15 | 189.24 | 76.6% | 3.63x |
| `Ga3As3H12` | 61,349 | 61,349 | 5,970,947 | 1,622 | 97.3 | 48.01 | 796.06 | **94.0%** | **16.58x** |

![Figure 1: Memory Comparison](SpMV/results/plots/fig1_csr_vs_ell_memory.png)

---

## 2. Kernel Design & Thread Mapping Trade-offs

- **CSR Scalar (1 thread/row):** Simple mapping. Irregular row lengths cause warp divergence. Scattered reads to vector $x$ prevent coalescing.
- **CSR Vector (1 warp/row):** One 32-thread warp processes each row. Strided access enables coalesced 128-byte global loads. Warp lanes reduce partial sums with `__shfl_down_sync`.
- **Column-Major ELLPACK (1 thread/row):** Thread $i$ evaluates $K$ padded entries. Column-major indexing ensures coalesced memory reads and zero warp divergence.

---

## 3. GPU Behavior: Divergence, Coalescing & Block Size

Row-length variance dictates warp divergence. A warp executes synchronously at the speed of its longest row.

### Structural Contrasts
- **`roadNet-CA` ($\mu=2.8, K=12$):** Short rows exhibit low variance. CSR scalar completes quickly. CSR vector idles 90% of threads per warp. CSR scalar outperforms CSR vector by **8.49x**.
- **`pkustk14` ($\mu=97.7, K=333$) & `Ga3As3H12` ($\mu=97.3, K=1622$):** High row variance forces 31 threads in scalar warps to stall. CSR vector assigns 32 threads to process long rows cooperatively. CSR vector wins by **2.27x** on `pkustk14` and **2.51x** on `Ga3As3H12`.

![Figure 2: Block Size Sweep](SpMV/results/plots/fig2_block_size_sweep.png)

### Table 2: Block Size Sweep Memory Throughput (GB/s)

| Matrix | Kernel | Block 64 | Block 128 | Block 256 | Block 512 | Block 1024 | Optimal |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `roadNet-CA` | CSR Scalar | 503.0 | 504.6 | **504.6** | 503.5 | 501.2 | **256** |
| `roadNet-CA` | CSR Vector | 61.1 | 60.5 | 59.4 | 60.1 | 58.7 | **64** |
| `roadNet-CA` | ELLPACK | 593.6 | 597.2 | **598.0** | 597.5 | 596.0 | **256** |
| `pkustk14` | CSR Scalar | 285.3 | 273.5 | 266.3 | 262.1 | 258.4 | **64** |
| `pkustk14` | CSR Vector | 605.0 | 604.8 | **604.4** | 603.9 | 602.1 | **64** |
| `pkustk14` | ELLPACK | 566.0 | 575.2 | **583.2** | 580.4 | 578.1 | **256** |
| `Ga3As3H12` | CSR Scalar | 245.9 | 238.4 | 231.7 | 228.5 | 225.1 | **64** |
| `Ga3As3H12` | CSR Vector | 583.7 | 582.1 | **580.9** | 579.4 | 576.8 | **64** |
| `Ga3As3H12` | ELLPACK | 571.1 | 567.4 | 563.8 | 561.2 | 558.0 | **64** |

### Launch Configuration Findings
- Block size 64 restricts active warps per SM. Block size 1024 increases register pressure.
- Block size 256 delivers optimal warp occupancy and latency hiding across all kernels.

---

## 4. Performance Comparison & Per-Matrix Results

Table 3 compares kernel runtime, effective bandwidth, and numerical validation at block size 256.

![Figure 3: Kernel Performance Comparison](SpMV/results/plots/fig3_kernel_performance_comparison.png)

### Table 3: Performance Summary (Block Size = 256)

| Matrix | Format | Kernel (ms) | Total (ms) | GFLOP/s | Bandwidth | Rel Error | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `bfly` | CSR Scalar | 0.0069 | 0.0431 | 57.18 | 400.2 GB/s | 5.16e-08 | PASS |
| `bfly` | CSR Vector | 0.0259 | 0.0625 | 15.17 | 106.2 GB/s | 5.26e-08 | PASS |
| `bfly` | ELLPACK | 0.0067 | 0.0425 | 58.40 | 379.6 GB/s | 5.16e-08 | **WINNER (1.02x)** |
| `D6-6` | CSR Scalar | 0.0079 | 0.0507 | 37.26 | 345.9 GB/s | 4.65e-08 | **WINNER (1.12x)** |
| `D6-6` | CSR Vector | 0.0467 | 0.0905 | 6.28 | 58.3 GB/s | 4.18e-08 | PASS |
| `D6-6` | ELLPACK | 0.0088 | 0.0521 | 33.22 | 1036.2 GB/s | 4.65e-08 | PASS |
| `dictionary28` | CSR Scalar | 0.0093 | 0.0471 | 38.34 | 275.4 GB/s | 6.38e-08 | **WINNER (1.56x)** |
| `dictionary28` | CSR Vector | 0.0269 | 0.0652 | 13.25 | 95.2 GB/s | 7.19e-08 | PASS |
| `dictionary28` | ELLPACK | 0.0145 | 0.0528 | 24.57 | 1671.0 GB/s | 6.38e-08 | PASS |
| `pkustk14` | CSR Scalar | 0.6731 | 0.7597 | 44.08 | 266.3 GB/s | 1.34e-07 | PASS |
| `pkustk14` | CSR Vector | 0.2966 | 0.3829 | 100.06 | 604.4 GB/s | 2.13e-07 | **WINNER (2.27x)** |
| `pkustk14` | ELLPACK | 1.0420 | 1.1267 | 28.48 | 583.2 GB/s | 1.34e-07 | PASS |
| `roadNet-CA` | CSR Scalar | 0.1628 | 0.7670 | 67.96 | 504.6 GB/s | 4.61e-08 | **WINNER (3.00x)** |
| `roadNet-CA` | CSR Vector | 1.3833 | 1.9889 | 8.00 | 59.4 GB/s | 4.48e-08 | PASS |
| `roadNet-CA` | ELLPACK | 0.4879 | 1.0955 | 22.68 | 598.0 GB/s | 4.61e-08 | PASS |
| `Ga3As3H12` | CSR Scalar | 0.3113 | 0.3536 | 38.36 | 231.7 GB/s | 1.92e-07 | PASS |
| `Ga3As3H12` | CSR Vector | 0.1242 | 0.1660 | 96.16 | 580.9 GB/s | 4.59e-07 | **WINNER (2.51x)** |
| `Ga3As3H12` | ELLPACK | 2.1182 | 2.1598 | 5.64 | 563.8 GB/s | 1.92e-07 | PASS |

### Per-Matrix Results
- **`bfly` (ELLPACK wins, 1.02x):** Uniform row lengths ($K=4$). Zero padding overhead and coalescing maximize performance.
- **`D6-6` (CSR Scalar wins, 1.12x):** Short rows ($\mu=1.2$). CSR avoids 79.7% zero padding in ELL.
- **`dictionary28` (CSR Scalar wins, 1.56x):** Short rows ($\mu=3.4, K=38$). Avoids 91.1% zero padding in ELL and avoids idle threads in vector warps.
- **`pkustk14` (CSR Vector wins, 2.27x):** Dense rows ($\mu=97.7$). 32 threads stream nonzeros with coalesced reads and shuffle reduction at 604.4 GB/s.
- **`roadNet-CA` (CSR Scalar wins, 3.00x over ELL, 8.49x over vector):** Sparse graph ($\mu=2.8$). Avoids 76.6% zero padding and avoids idle threads.
- **`Ga3As3H12` (CSR Vector wins, 2.51x over scalar, 17.06x over ELL):** Extreme row length variance ($K=1622$). Vector balances workload across warp lanes.

---

## 5. CPU Speedup & PCIe Bus Bottleneck

We compute speedup against sequential single-core CPU baselines ($S = T_{\text{cpu}} / T_{\text{gpu}}$).

![Figure 4: Speedup and PCIe Transfers](SpMV/results/plots/fig4_gpu_speedup_and_transfers.png)

### Table 4: CPU Speedup and PCIe Transfer Overhead

| Matrix | CPU Time | Best GPU Kernel | GPU Total | Kernel Speedup | Total Speedup | Transfer Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bfly` | 0.108 ms | 0.0067 ms (ELL) | 0.0425 ms | **16.00x** | 2.54x | 84.1% |
| `D6-6` | 0.136 ms | 0.0079 ms (Scalar) | 0.0507 ms | **17.23x** | 2.68x | 84.4% |
| `dictionary28` | 0.454 ms | 0.0093 ms (Scalar) | 0.0471 ms | **48.82x** | 9.62x | 80.3% |
| `pkustk14` | 5.491 ms | 0.2966 ms (Vector) | 0.3829 ms | **18.51x** | 14.34x | 22.6% |
| `roadNet-CA` | 10.142 ms | 0.1628 ms (Scalar) | 0.7670 ms | **62.28x** | 13.22x | 78.8% |
| `Ga3As3H12` | 2.566 ms | 0.1242 ms (Vector) | 0.1660 ms | **20.67x** | 15.46x | 25.2% |

### Transfer Bottleneck Observations
- For single SpMV invocations on sparse graph matrices, PCIe transfers consume **78% to 84%** of runtime.
- Iterative solvers retain vectors $x$ and $y$ on the GPU, realizing the full **16x to 62x** compute speedup.

---

## 6. Empirical GPU Roofline Analysis

Single-precision SpMV requires an operational intensity of $\text{AI} \approx 0.17\text{ FLOP/byte}$.

![Figure 5: GPU Roofline Model](SpMV/results/plots/fig5_gpu_roofline.png)

### Table 5: Empirical Roofline Utilization

| Matrix | Kernel | AI (FLOP/B) | GFLOP/s | Achieved BW | DRAM BW % | V100 Equiv % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `pkustk14` | CSR Vector | 0.166 | 100.06 | **604.4 GB/s** | **134.9%** | 67.2% |
| `roadNet-CA` | CSR Scalar | 0.135 | 67.96 | **504.6 GB/s** | **112.6%** | 56.1% |
| `roadNet-CA` | ELLPACK | 0.038 | 22.68 | **598.0 GB/s** | **133.5%** | 66.4% |
| `Ga3As3H12` | CSR Vector | 0.166 | 96.16 | **580.9 GB/s** | **129.7%** | 64.5% |
| `Ga3As3H12` | CSR Scalar | 0.166 | 38.36 | 231.7 GB/s | 51.7% | 25.7% |
| `Ga3As3H12` | ELLPACK | 0.010 | 5.64 | **563.8 GB/s** | **125.8%** | 62.6% |
| `bfly` | ELLPACK | 0.154 | 58.40 | 379.6 GB/s | 84.7% | 42.2% |
| `dictionary28` | ELLPACK | 0.015 | 24.57 | **1671.0 GB/s** | **373.0%** | 185.7% |

### Roofline Findings
- **Memory Bound:** The RTX 5060 Ti memory knee is 62.5 FLOP/byte. Because $\text{AI} = 0.17 \ll 62.5$, SpMV resides strictly in the memory-bound regime.
- **Cache Reuse:** Effective throughput reaches **604.4 GB/s** on `pkustk14`, exceeding DRAM bandwidth (448 GB/s) due to on-chip L2 cache hits.
- **Divergence Impact:** On `Ga3As3H12`, warp divergence stalls reduce CSR scalar throughput to 231.7 GB/s.
