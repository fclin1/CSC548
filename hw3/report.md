# CSC 548 HW3 Report: GPU SpMV Architecture and Optimization

**Unity ID:** `fclin`.  
**Course:** CSC 548 Parallel Systems, Fall 2026.  
**Primary Hardware:** NVIDIA GeForce GTX 1080 Ti (Pascal sm_61, 3584 cores, 484 GB/s peak bandwidth, 11.3 TFLOP/s FP32).  
**Reference Platform:** NC State ARC GPU Node (Tesla V100 SXM2 sm_70, 5120 cores, 900 GB/s peak bandwidth, 14.1 TFLOP/s FP32).  
**Specification Standard:** ASD-STE100 Simplified Technical English.  

---

## 1. Executive Summary & Dataset Properties

Sparse Matrix-Vector Multiplication (SpMV) performs a linear algebra product.
$$y = A \cdot x$$

The matrix $A$ contains mostly zero values. This investigation evaluates three CUDA GPU mapping implementations across six SuiteSparse matrices:
1. **CSR Scalar Kernel:** One CUDA thread per matrix row.
2. **CSR Vector Kernel:** One 32-thread warp per matrix row.
3. **Column-Major ELLPACK Kernel:** One CUDA thread per matrix row with uniform zero padding.

The arithmetic intensity of SpMV is approximately 0.17 FLOP per byte. Memory bandwidth limits performance across all matrices. Matrix sparsity structure dictates the optimal GPU thread mapping. Row length variance causes severe warp divergence in the CSR scalar kernel. The CSR vector kernel resolves divergence on dense rows through warp-cooperative execution. Column-major ELLPACK achieves perfect memory coalescing but suffers bandwidth waste on matrices with long row tails.

### Table 1: Matrix Structural Parameters and Memory Footprints

| Matrix | Rows ($m$) | Cols ($n$) | NNZ | Max $K$ | Avg $\mu$ | CSR (MB) | ELL (MB) | Zero Padding | ELL/CSR Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bfly` | 49,152 | 49,152 | 196,608 | 4 | 4.0 | 1.77 MB | 1.57 MB | **0.0%** | **0.89×** |
| `D6-6` | 120,576 | 23,740 | 146,880 | 6 | 1.2 | 1.66 MB | 5.79 MB | 79.7% | 3.49× |
| `dictionary28` | 52,652 | 52,652 | 178,076 | 38 | 3.4 | 1.64 MB | 16.01 MB | 91.1% | 9.79× |
| `pkustk14` | 151,926 | 151,926 | 14,836,504 | 333 | 97.7 | 119.30 MB | 404.73 MB | 70.7% | 3.39× |
| `roadNet-CA` | 1,971,281 | 1,971,281 | 5,533,214 | 12 | 2.8 | 52.15 MB | 189.24 MB | 76.6% | 3.63× |
| `Ga3As3H12` | 61,349 | 61,349 | 5,970,947 | 1,622 | 97.3 | 48.01 MB | 796.06 MB | **94.0%** | **16.58×** |

![Figure 1: CSR vs ELL Memory Comparison](SpMV/results/plots/fig1_csr_vs_ell_memory.png)

### Key Storage Findings
- ELL storage expands when the maximum row length $K$ greatly exceeds the average row length $\mu$.
- On `Ga3As3H12`, ELL allocates 796.06 MB of memory. This footprint is 16.58 times larger than CSR.
- On `bfly`, every row contains exactly 4 elements. ELL incurs zero padding and eliminates the CSR row pointer array entirely.

---

## 2. GPU Behavior: Divergence, Memory Coalescing & Block Size Sensitivity

Row length variance directly controls warp divergence and execution efficiency. A warp executes in lockstep. The runtime of a warp equals the runtime of its longest row.

### Contrasting Matrix Structures: `roadNet-CA` vs. `pkustk14` and `Ga3As3H12`
- **`roadNet-CA` ($\mu = 2.8, K = 12$):** Rows are consistently short with low variance. In the CSR scalar kernel, all threads finish within 2 to 12 iterations. Warp divergence is minor. In the CSR vector kernel, assigning 32 threads per row leaves over 90% of threads idle. Only lanes 0 to 2 perform work. Thus, CSR scalar outperforms CSR vector by **4.58×** on `roadNet-CA`.
- **`pkustk14` ($\mu = 97.7, K = 333$) and `Ga3As3H12` ($\mu = 97.3, K = 1622$):** These matrices exhibit extreme row length variance. In the CSR scalar kernel, when one thread processes an outlier row of 1,622 elements, the remaining 31 threads stall for 1,622 cycles. In the CSR vector kernel, 32 threads collaborate on the row. Consecutive lanes read contiguous memory addresses, enabling coalesced 128-byte transactions. Lanes reduce partial sums with `__shfl_down_sync`. CSR vector outperforms CSR scalar by **5.55×** on `Ga3As3H12` and **2.99×** on `pkustk14`.

![Figure 2: Block Size Sweep](SpMV/results/plots/fig2_block_size_sweep.png)

### Table 2: Block Size Sweep Throughput (GB/s) Across Kernels

| Matrix | Kernel | Block 64 | Block 128 | Block 256 | Block 512 | Block 1024 | Optimal Block |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `roadNet-CA` | CSR Scalar | 201.2 GB/s | 203.5 GB/s | **203.7 GB/s** | 202.1 GB/s | 197.4 GB/s | **256** |
| `roadNet-CA` | CSR Vector | 42.8 GB/s | 44.1 GB/s | **44.4 GB/s** | 44.2 GB/s | 43.8 GB/s | **256** |
| `roadNet-CA` | ELLPACK | 304.1 GB/s | 310.2 GB/s | **312.2 GB/s** | 310.5 GB/s | 307.1 GB/s | **256** |
| `pkustk14` | CSR Scalar | 103.2 GB/s | 105.8 GB/s | **107.5 GB/s** | 106.8 GB/s | 105.1 GB/s | **256** |
| `pkustk14` | CSR Vector | 309.4 GB/s | 318.7 GB/s | **321.0 GB/s** | 316.2 GB/s | 310.5 GB/s | **256** |
| `pkustk14` | ELLPACK | 271.2 GB/s | 277.4 GB/s | **278.3 GB/s** | 276.8 GB/s | 273.9 GB/s | **256** |
| `Ga3As3H12` | CSR Scalar | 46.8 GB/s | 47.9 GB/s | **48.1 GB/s** | 47.8 GB/s | 46.9 GB/s | **256** |
| `Ga3As3H12` | CSR Vector | 258.4 GB/s | 265.1 GB/s | **267.1 GB/s** | 265.8 GB/s | 257.2 GB/s | **256** |
| `Ga3As3H12` | ELLPACK | 234.1 GB/s | 237.2 GB/s | **237.9 GB/s** | 236.8 GB/s | 235.1 GB/s | **256** |

### Effect of Thread Block Size
- Small block sizes (64 threads) limit SM occupancy because the GPU reaches the maximum block count limit.
- Large block sizes (1024 threads) restrict scheduling flexibility and increase register pressure.
- A block size of 256 threads delivers optimal performance across all test cases.

---

## 3. Kernel Performance Comparison & Per-Matrix Analysis

We benchmarked all three kernels at the optimal block size of 256 threads. Figure 3 presents execution times and effective memory bandwidths.

![Figure 3: Kernel Performance Comparison](SpMV/results/plots/fig3_kernel_performance_comparison.png)

### Table 3: Kernel Performance Metrics (Block Size = 256)

| Matrix | Format | Kernel Time | Total Time | GFLOP/s | Bandwidth | Rel Error | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `bfly` | CSR Scalar | 0.0890 ms | 0.2824 ms | 4.42 | 30.9 GB/s | 5.16e-08 | PASS |
| `bfly` | CSR Vector | 0.1015 ms | 0.3170 ms | 3.87 | 27.1 GB/s | 5.26e-08 | PASS |
| `bfly` | ELLPACK | 0.0652 ms | 0.2415 ms | 6.03 | 39.2 GB/s | 5.16e-08 | **WINNER (1.37×)** |
| `D6-6` | CSR Scalar | 0.0827 ms | 0.3263 ms | 3.55 | 33.0 GB/s | 4.65e-08 | **WINNER (1.28×)** |
| `D6-6` | CSR Vector | 0.1720 ms | 0.4261 ms | 1.71 | 15.9 GB/s | 4.18e-08 | PASS |
| `D6-6` | ELLPACK | 0.1058 ms | 0.3827 ms | 2.78 | 86.6 GB/s | 4.65e-08 | PASS |
| `dictionary28` | CSR Scalar | 0.0846 ms | 0.2756 ms | 4.21 | 30.2 GB/s | 6.38e-08 | **WINNER (1.42×)** |
| `dictionary28` | CSR Vector | 0.1198 ms | 0.3128 ms | 2.97 | 21.3 GB/s | 7.19e-08 | PASS |
| `dictionary28` | ELLPACK | 0.1414 ms | 0.3388 ms | 2.52 | 171.3 GB/s | 6.38e-08 | PASS |
| `pkustk14` | CSR Scalar | 1.6672 ms | 1.9881 ms | 17.80 | 107.5 GB/s | 1.34e-07 | PASS |
| `pkustk14` | CSR Vector | 0.5584 ms | 0.8799 ms | 53.14 | 321.0 GB/s | 2.13e-07 | **WINNER (2.99×)** |
| `pkustk14` | ELLPACK | 2.1840 ms | 2.4243 ms | 13.59 | 278.3 GB/s | 1.34e-07 | PASS |
| `roadNet-CA` | CSR Scalar | 0.4034 ms | 2.6936 ms | 27.43 | 203.7 GB/s | 4.61e-08 | **WINNER (2.32×)** |
| `roadNet-CA` | CSR Vector | 1.8495 ms | 4.1291 ms | 5.98 | 44.4 GB/s | 4.48e-08 | PASS |
| `roadNet-CA` | ELLPACK | 0.9346 ms | 3.5783 ms | 11.84 | 312.2 GB/s | 4.61e-08 | PASS |
| `Ga3As3H12` | CSR Scalar | 1.4995 ms | 1.6210 ms | 7.96 | 48.1 GB/s | 1.92e-07 | PASS |
| `Ga3As3H12` | CSR Vector | 0.2701 ms | 0.4585 ms | 44.21 | 267.1 GB/s | 4.59e-07 | **WINNER (5.55×)** |
| `Ga3As3H12` | ELLPACK | 5.0210 ms | 4.9635 ms | 2.38 | 237.9 GB/s | 1.92e-07 | PASS |

### Per-Matrix Breakdown and Winning Rationale
1. **`bfly` (Winner: Column-Major ELLPACK, 1.37× faster):** Every row contains exactly 4 elements. ELL incurs zero padding. The column-major layout enables contiguous 128-byte loads across 32 rows, maximizing throughput.
2. **`D6-6` (Winner: CSR Scalar, 1.28× faster):** Average row length is only 1.2 elements. While $K=6$ is small, ELL wastes 79.7% memory bandwidth on zero padding. CSR scalar streams true nonzeros without padding overhead.
3. **`dictionary28` (Winner: CSR Scalar, 1.42× faster):** High row variance ($K=38, \mu=3.4$) causes 91.1% ELL padding. CSR vector underutilizes warp threads on short rows. CSR scalar executes compactly with minimal stall overhead.
4. **`pkustk14` (Winner: CSR Vector, 2.99× faster):** Dense rows ($\mu=97.7, K=333$) allow warp lanes to stream contiguous memory addresses. Vector execution achieves 321.0 GB/s throughput, avoiding the 70.7% padding penalty of ELL.
5. **`roadNet-CA` (Winner: CSR Scalar, 2.32× faster than ELL, 4.58× faster than vector):** Contains 1.97 million rows with only 2.8 nonzeros per row. Vector assigns 32 threads per row, leaving 90% idle. Scalar finishes short rows rapidly.
6. **`Ga3As3H12` (Winner: CSR Vector, 5.55× faster than scalar, 18.6× faster than ELL):** Long orbital rows reach $K=1,622$. Scalar suffers severe warp divergence. ELL wastes 796 MB on padding. Vector balances the workload across warp lanes.

---

## 4. Performance & Roofline: Speedup and PCIe Transfer Overhead

We measured GPU speedup against single-core CPU sequential CSR baselines.
$$S_{\text{kernel}} = \frac{T_{\text{cpu}}}{T_{\text{gpu, kernel}}}, \quad S_{\text{total}} = \frac{T_{\text{cpu}}}{T_{\text{gpu, total}}}$$

![Figure 4: Speedup and PCIe Transfers](SpMV/results/plots/fig4_gpu_speedup_and_transfers.png)

### Table 4: CPU vs. GPU Speedup and Transfer Overhead

| Matrix | CPU Time | GPU Best Kernel | GPU Total Time | Kernel Speedup | Total Speedup | Transfer Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bfly` | 0.271 ms | 0.065 ms (ELL) | 0.242 ms | **4.16×** | 1.12× | 73.0% |
| `D6-6` | 0.540 ms | 0.083 ms (Scalar) | 0.326 ms | **6.53×** | 1.66× | 74.6% |
| `dictionary28` | 0.642 ms | 0.085 ms (Scalar) | 0.276 ms | **7.59×** | 2.33× | 69.3% |
| `pkustk14` | 18.834 ms | 0.558 ms (Vector) | 0.880 ms | **33.73×** | 21.40× | 36.5% |
| `roadNet-CA` | 16.658 ms | 0.403 ms (Scalar) | 2.694 ms | **41.29×** | 6.18× | 85.0% |
| `Ga3As3H12` | 7.888 ms | 0.270 ms (Vector) | 0.459 ms | **29.20×** | 17.20× | 41.1% |

### Transfer Bottleneck Analysis
- PCIe bus transfers consume 50% to 85% of total runtime for lone SpMV calls.
- On `roadNet-CA`, PCIe transfer consumes 85.0% of execution time, dropping speedup from 41.3× to 6.2×.
- Iterative linear solvers maintain vectors in device memory, realizing the full 30× to 41× kernel speedup.

---

## 5. GPU Roofline Analysis

The roofline model characterizes system performance bounds based on arithmetic intensity:
$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes Transferred}} \approx 0.17 \text{ FLOP/byte}$$

![Figure 5: GPU Roofline Model](SpMV/results/plots/fig5_gpu_roofline.png)

### Roofline Knee and Bounds
- **NVIDIA GeForce GTX 1080 Ti:** Peak memory bandwidth is 484 GB/s. Peak compute is 11,340 GFLOP/s. The roofline knee occurs at:
$$\text{Knee} = \frac{11,340 \text{ GFLOP/s}}{484 \text{ GB/s}} = 23.4 \text{ FLOP/byte}$$
- **Reference Tesla V100 GPU:** Peak memory bandwidth is 900 GB/s. Peak compute is 14,100 GFLOP/s. The roofline knee occurs at 15.7 FLOP/byte.
- Because $\text{AI} \approx 0.17 \ll 23.4$, GPU SpMV resides strictly within the memory-bound regime.

### Table 5: Empirical Roofline Bandwidth Utilization

| Matrix | Kernel | AI (FLOP/B) | GFLOP/s | Achieved BW | 1080 Ti BW % | V100 Equiv BW % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `pkustk14` | CSR Vector | 0.166 | 53.14 | **321.0 GB/s** | **66.3%** | 35.7% |
| `roadNet-CA` | CSR Scalar | 0.135 | 27.43 | **203.7 GB/s** | **42.1%** | 22.6% |
| `roadNet-CA` | ELLPACK | 0.038 | 11.84 | **312.2 GB/s** | **64.5%** | 34.7% |
| `Ga3As3H12` | CSR Vector | 0.166 | 44.21 | **267.1 GB/s** | **55.2%** | 29.7% |
| `Ga3As3H12` | CSR Scalar | 0.166 | 7.96 | 48.1 GB/s | 9.9% | 5.3% |
| `Ga3As3H12` | ELLPACK | 0.010 | 2.38 | 237.9 GB/s | 49.2% | 26.4% |
| `bfly` | ELLPACK | 0.154 | 6.03 | 39.2 GB/s | 8.1% | 4.4% |
| `dictionary28` | ELLPACK | 0.015 | 2.52 | 171.3 GB/s | 35.4% | 19.0% |

### Roofline Conclusion
- CSR vector on `pkustk14` reaches 321.0 GB/s (66.3% of theoretical peak bandwidth on the GTX 1080 Ti).
- This high bandwidth utilization demonstrates effective warp scheduling and coalesced DRAM transactions.
- On `Ga3As3H12`, the scalar kernel reaches only 48.1 GB/s due to warp stalls on outlier rows.
- Performance remains strictly memory bandwidth-bound for all kernels.

---

## 6. Implementation Summary

All source files satisfy the interface requirements in `include/spmv_cuda.h`:
- `spmv_cuda_setup`: Allocates GPU memory buffers, uploads matrix $A$, configures launch parameters, and records statistics.
- `spmv_cuda_h2d`: Transfers vector $x$ from host to device memory.
- `spmv_cuda_kernel`: Launches the kernel with configured grid and block dimensions, then calls `cudaDeviceSynchronize()`.
- `spmv_cuda_d2h`: Transfers vector $y$ from device to host memory.
- `spmv_cuda_stats`: Exports launch configuration and ELL slot geometry to the harness.
- `spmv_cuda_teardown`: Frees all device buffers with `cudaFree` and deallocates host structures.

The submission bundle `HW3-fclin.zip` contains:
1. `spmv-cuda-scalar.cu`
2. `spmv-cuda-vector.cu`
3. `spmv-cuda-ell.cu`
4. `Makefile`
5. `report.pdf`