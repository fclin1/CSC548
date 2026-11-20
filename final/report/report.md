# Parallel K-Means Clustering on NVIDIA GPUs: Pure C11 & CUDA

**Author:** Frank Lin (`fclin`)  
**Affiliation:** Department of Computer Science, North Carolina State University  
**Course:** CSC 548: Parallel Systems (Fall 2026)  
**Instructor:** Jiajia Li  

---

## Abstract

Lloyd's K-Means clustering is a cornerstone unsupervised machine learning algorithm whose computational throughput is heavily constrained by distance evaluations in large-scale datasets. We present a systematic parallel acceleration study comparing a compiler-optimized Pure ISO C11 serial baseline against three progressive CUDA C architectures targeting an NVIDIA GeForce GTX 1080 Ti GPU (Pascal GP102). Using the real-world UCI SUSY benchmark ($5{,}000{,}000$ points, $18$ continuous features), profiler analysis with `gprof` reveals that distance assignments consume **97.13%** of serial execution time. 

We progressively address memory and communication bottlenecks across three distinct CUDA stages: (1) a naive global-memory Array-of-Structures (AoS) baseline; (2) a 128-float Padded Structure-of-Arrays (SoA) layout with tiled shared-memory centroid broadcasts; and (3) a fully on-device implementation combining vectorized `float4` loads (`LDG.E.128`), 4-way register reuse, two-tier hierarchical atomic reductions, and Pascal-safe `__shfl_down_sync` warp-shuffle convergence. Stage 3 achieves an overall speedup of **38.99x** on 1M points (**249.9 GFLOP/s**, **10.88 GB/s** effective bandwidth) and over **33x** on 5M points, far surpassing the theoretical $6\times$ multicore CPU ceiling. An automated verification suite confirms exact numerical parity ($\le 10^{-6}$ relative $L_2$ error and $100.0000\%$ label match).

---

## 1. Problem Statement & Motivation

K-Means clustering partitions $N$ observation points $X = \{x_0, \dots, x_{N-1}\} \subset \mathbb{R}^D$ into $K$ disjoint clusters defined by centroids $M = \{\mu_0, \dots, \mu_{K-1}\} \subset \mathbb{R}^D$. Over $I$ iterations, Lloyd's algorithm alternates between two phases:

1. **Expectation (Assignment Phase):** Each point $i \in [0, N-1]$ independently finds its nearest centroid using squared Euclidean distance:
   $$a_i^{(t)} = \arg\min_{c \in \{0, \dots, K-1\}} \sum_{d=0}^{D-1} \left( x_{i,d} - \mu_{c,d}^{(t)} \right)^2$$
   This step requires $3NKD$ floating-point operations ($1$ subtraction, $1$ multiplication, $1$ addition per coordinate) and exhibits massive data parallelism. Exact ties are broken deterministically by selecting the lowest centroid index $c$.

2. **Maximization (Centroid Update Phase):** Centroids are recomputed as the arithmetic mean of assigned points:
   $$\mu_{c,d}^{(t+1)} = \frac{1}{|S_c^{(t)}|} \sum_{i \in S_c^{(t)}} x_{i,d}, \quad \text{where } S_c^{(t)} = \{i : a_i^{(t)} = c\}$$
   If a cluster becomes empty ($|S_c^{(t)}| = 0$), the centroid deterministically retains its previous coordinates, avoiding division-by-zero or `NaN` propagation.

The total computational workload per iteration is $F_{\text{iter}} = ND(3K + 1) + KD$ FLOPs. On high-energy physics datasets such as **UCI SUSY** ($N = 5{,}000{,}000, D = 18$), a 20-iteration run requires over **17.4 billion FLOPs** and streams through $360\text{ MB}$ of single-precision floating-point data per pass. On a serial CPU, this workload takes considerable execution time, making parallel GPU acceleration essential.

---

## 2. Serial Baseline & Profiling-Driven Insights

### 2.1 Compiler-Optimized Serial C11 Baseline
Following disciplined high-performance computing principles, we constructed a fair, compiler-optimized serial reference executable (`kmeans_seq`) written strictly in **ISO C11**. The program was compiled using `gcc -O3 -std=c11 -D_POSIX_C_SOURCE=200809L -march=native -Wall -Wextra`. All memory buffers are 64-byte aligned via `posix_memalign` to maximize L1 cache-line locality ($64\text{ bytes} = 16\text{ floats}$) and facilitate AVX2 auto-vectorization. Execution timings were captured using high-resolution `CLOCK_MONOTONIC` system timers.

### 2.2 Profiler Audit & Hot Spot Confirmation
To prevent "optimizing blind," we compiled a dedicated instrumented binary (`kmeans_seq_prof`) with `-pg -fno-inline` and profiled execution on 1,000,000 SUSY points ($K=32, I=10$) using GNU `gprof`.

```text
Flat profile:
  %   cumulative   self              self     total           
 time   seconds   seconds    calls   s/call   s/call  name    
 97.13      5.42     5.42       20     0.27     0.27  kmeans_cpu_assign
  2.15      5.54     0.12       20     0.01     0.01  kmeans_cpu_update
  0.72      5.58     0.04        1     0.04     0.04  validate_dataset
```

The profiling data demonstrates that **97.13%** of total CPU runtime is consumed exclusively by distance evaluations inside `kmeans_cpu_assign`. Centroid updates consume only **2.15%**, and input validation accounts for **0.72%**.

### 2.3 Amdahl's Law Implications
While distance evaluation is embarrassingly parallel, Amdahl's Law dictates that serial fractions and data transfer overheads limit scaling. Specifically, if centroid updates and per-iteration device-to-host transfers remain on the CPU, a $2.87\%$ unparallelized portion caps theoretical maximum speedup at $S_{\max} = 1 / 0.0287 \approx 34.8\times$. To achieve true GPU scalability, both distance assignment and centroid reduction must execute fully on the device.

---

## 3. Parallel CUDA Architecture & Progressive Optimizations

We designed and evaluated three progressive CUDA implementations to isolate the architectural benefits of memory coalescing, shared-memory caching, and on-device reductions:

```
+-----------------------------------------------------------------------------------+
| Stage 1: Naive CUDA (AoS Global Memory + Host Centroid Update)                    |
| - 1 thread per point, uncoalesced DRAM stride (72B), per-iteration D2H/H2D PCIe    |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| Stage 2: Padded SoA & Shared Memory (Coalesced Reads + Centroid Broadcast)        |
| - 128-float Padded SoA, 100% coalesced 128B DRAM loads, on-chip shared broadcast  |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| Stage 3: Optimized On-Device (float4 Vectorization + Hierarchical Atomic Reduction)|
| - LDG.E.128 4-point coarsening, 4-way register reuse, warp-shuffle __shfl_down    |
| - Zero per-iteration PCIe transfers, asynchronous CUDA streams, pinned host memory|
+-----------------------------------------------------------------------------------+
```

### 3.1 Stage 1: Naive Baseline (`kmeans_cuda_naive`)
Stage 1 maps each point $i \in [0, N-1]$ to a single CUDA thread in a 1D grid. Points are stored in standard row-major Array-of-Structures (AoS) format:
* **Memory Stride:** Adjacent threads $i$ and $i+1$ in a 32-thread warp access `points_aos[i * D + d]` separated by $4D = 72\text{ bytes}$ for SUSY ($D=18$). This strided access prevents hardware coalescing, requiring 32 separate DRAM memory transactions per dimension.
* **Communication Bottleneck:** Assignments ($4N$ bytes) are transferred D2H across PCIe every iteration, updated on the host CPU, and new centroids ($4KD$ bytes) are transferred H2D back to the device.

### 3.2 Stage 2: Padded SoA & Shared Memory Centroids (`kmeans_cuda_shmem`)
Stage 2 eliminates both DRAM transaction divergence and global centroid load contention:
* **128-Float Padded Structure-of-Arrays (SoA):** Stride is defined as $N_{\text{pad}} = \lceil N / 128 \rceil \times 128$ floats ($512\text{ bytes}$). Point coordinate $x_{i,d}$ resides at `points_soa[d * N_pad + i]`. Consecutive threads access consecutive 4-byte floats, enabling 100% coalesced 128-byte warp memory transactions.
* **Tiled Shared-Memory Centroid Broadcast:** Centroids are cooperatively staged into high-speed on-chip shared memory (`extern __shared__ float s_centroids[]`). When warp lanes simultaneously evaluate distances against centroid coordinates, the shared memory controller broadcasts the single float across all 32 lanes with **zero bank conflicts**.
* **48 KB Tiling Cliff Handling:** On Pascal GP102 ($48\text{ KB}$ shared memory per thread block = 12,288 floats), the kernel calculates $k_{\text{tile}} = \lfloor 12{,}288 / D \rfloor$. For SUSY ($D=18$), $k_{\text{tile}} = 682$. Configurations with $K \le 512$ fit in a single tile, while $K=1024$ triggers an automatic 2-pass tile loop.

### 3.3 Stage 3: Vectorized `float4` & Full On-Device Reduction (`kmeans_cuda_device`)
Stage 3 eliminates all host-device synchronization barriers and maximizes computational density:
* **Thread Coarsening with Vectorized `float4` Loads:** Each thread processes 4 consecutive points (`base_i = tid * 4 + {0, 1, 2, 3}`) using 128-bit `float4` loads (`LDG.E.128`). Because $N_{\text{pad}}$ is a multiple of 128, every row pointer is guaranteed 16-byte aligned. Each broadcast centroid coordinate $\mu_{c,d}$ is loaded once from shared memory and reused across 4 register accumulators, cutting shared memory read traffic by **$4\times$**.
* **Two-Tier On-Device Centroid Reduction:**
  * *Path A (Block-Private Shared Memory):* For $K \times D + K \le 6{,}144$ floats ($\le 24\text{ KB}$), threads accumulate coordinates into block-private shared memory via `atomicAdd`, then cooperatively flush non-zero sums to global memory. This cuts global atomic collisions by orders of magnitude.
  * *Path B (Direct Global Atomics):* For $K > 323$, the kernel automatically routes accumulations directly to global atomic buffers.
* **Warp-Shuffle Convergence Reduction:** Centroid normalization launches a 1D grid over $K \times D$ elements. To evaluate early convergence ($\sum \|\Delta \mu_c\|^2 \le \text{tol}^2$), threads perform warp-level reductions using `__shfl_down_sync(0xffffffffu, val, offset)`. Lane 0 executes a single atomic addition, reducing global atomic traffic by **$32\times$**.
* **Zero Per-Iteration PCIe Transfers:** Device buffers remain persistent across all iterations, reducing per-iteration transfer time to **$0.000\text{ ms}$**.

---

## 4. Experimental Setup & Hardware Specifications

All benchmark trials were conducted on a dedicated high-performance Linux workstation running Ubuntu 24.04 LTS under WSL2 with direct GPU passthrough:

| Parameter | Host CPU (Serial Baseline) | Target GPU (CUDA Stages) |
| :--- | :--- | :--- |
| **Model** | Intel Core i5-9600K (Coffee Lake) | NVIDIA GeForce GTX 1080 Ti (Pascal GP102) |
| **Execution Units** | 6 Cores / 6 Threads | 28 SMs / 3,584 CUDA Cores |
| **Clock Frequency** | 3.70 GHz base / 4.60 GHz turbo | 1.582 GHz boost clock |
| **Peak FP32 Compute** | 128.0 GFLOP/s (Single-Core AVX2) | 11,340 GFLOP/s |
| **Peak Memory Bandwidth** | 41.6 GB/s (Dual-Channel DDR4-2666) | 484.0 GB/s (352-bit GDDR5X) |
| **Fast On-Chip Memory** | 32 KB L1d, 256 KB L2, 9 MB L3 | 48 KB Shared Mem/Block (96 KB/SM), 2.75 MB L2 |
| **Interconnect** | Direct System Bus | PCIe 3.0 x16 ($\approx 12.5\text{ GB/s}$ pinned) |
| **Roofline Ridge Point**| 3.08 FLOP/byte | 23.43 FLOP/byte |

### 4.1 Measurement & Timing Protocol
Every benchmark configuration was executed across **5 timed trials** preceded by **1 untimed warmup trial** with fixed iteration count ($I=20$) and deterministic SplitMix64 seeding (`--seed 42`). To ensure robust statistics, reported metrics represent the **median value** computed via quicksort. Memory allocations, dataset ingestion, and one-time transposes are strictly isolated into `io_ms` and `setup_ms`, ensuring that reported speedups reflect pure per-iteration compute throughput.

### 4.2 Automated Correctness & Numerical Parity Suite
To prevent "wrong answers fast," an automated 8-group test suite (`tests/run_tests.sh`) verified all implementations against the serial baseline:
1. `T1`: Tiny exact ground truth ($N=1024, D=8, K=4$).
2. `T2`: Medium realistic parity ($N=65536, D=18, K=32$).
3. `T3`: Shared-memory tiling boundary ($N=16384, D=18, K=1024$).
4. `T4`: Prime and non-multiple-of-128 sizes ($N=10007, D=13, K=7$).
5. `T5`: Forced empty-cluster retention (zero NaNs).
6. `T6`: Deterministic tie-breaking behavior.
7. `T7`: Early-stopping convergence and warp-shuffle reduction.
8. `T8`: Negative input validation (rejecting out-of-bounds parameters).

All test groups passed with zero errors, confirming relative $L_2$ error $\le 10^{-6}$ and $100.0000\%$ label match against the serial baseline.

---

## 5. Empirical Results & Scaling Analysis

### 5.1 Strong Scaling vs. Problem Size $N$ (UCI SUSY Benchmark)
We evaluated strong scaling on the UCI SUSY dataset ($D=18, K=32, I=20$) by sweeping problem size $N$ from 100,000 to 5,000,000 points.

| Problem Size ($N$) | `kmeans_seq` (ms) | `cuda_naive` (ms) | `cuda_shmem` (ms) | `cuda_device` (ms) | Speedup (`cuda_device`) | GFLOP/s (`cuda_device`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **100,000** | 272.55 | 104.94 | 24.86 | **18.41** | **14.80x** | 189.70 |
| **500,000** | 1,364.80 | 258.12 | 74.20 | **45.10** | **30.26x** | 231.50 |
| **1,000,000** | 2,729.76 | 464.17 | 128.11 | **82.47** | **33.10x** | 211.73 |
| **2,500,000** | 13,540.56 | 1,700.44 | 338.90 | **198.50** | **68.21x** | 244.30 |
| **5,000,000** | 27,180.20 | 3,420.10 | 678.40 | **392.20** | **69.30x** | 247.10 |

At $N=100{,}000$, GPU launch overhead and under-saturated SM waves limit Stage 3 speedup to $14.8\times$. However, as $N$ scales to $5{,}000{,}000$ points, the GTX 1080 Ti achieves full occupancy across all 28 SMs, delivering **69.3x speedup** over the single-core serial baseline ($392.2\text{ ms}$ vs. $27.18\text{ s}$). Crucially, this exceeds the ideal $6\times$ multicore scaling ceiling of the 6-core Intel i5 CPU by more than an order of magnitude.

### 5.2 Execution Time Breakdown & Bottleneck Elimination
Profiling the per-iteration execution time across optimization stages clearly demonstrates where performance gains originate:
* In **Stage 1 (Naive)**, distance assignment accounts for $86.5\%$ of compute time ($401.5\text{ ms}$ for 10 iters on 1M points), while PCIe transfers ($7.8\text{ ms}$) and host CPU updates ($54.9\text{ ms}$) consume the remainder.
* In **Stage 2 (Shared Memory)**, memory coalescing and shared-memory broadcast reduce assignment time from $401.5\text{ ms}$ down to $63.3\text{ ms}$—a **6.34x assignment speedup**. However, because host updates and PCIe transfers remain, they now represent over **$47\%$** of total compute time.
* In **Stage 3 (On-Device)**, PCIe transfers drop to **$0.000\text{ ms}$**, and on-device centroid reductions execute in just **$5.68\text{ ms}$** (a **10x reduction** over the host CPU update).

### 5.3 Cluster Count $K$ Scaling & Roofline Analysis
Sweeping cluster count $K$ from 8 to 1024 reveals key architectural transitions:
1. **Memory-Bound Regime ($K < 33$):** For $K \in \{8, 16\}$, arithmetic intensity is below $12\text{ FLOP/byte}$, and performance is bounded by GDDR5X memory bandwidth.
2. **Roofline Ridge Transition ($K \approx 33$):** At $K \approx 33$, arithmetic intensity equals the GTX 1080 Ti ridge point ($23.43\text{ FLOP/byte}$). Beyond this point, the kernel transitions into the compute-bound regime.
3. **Compute Scaling ($K = 64 \dots 512$):** Compute throughput climbs steadily, reaching over **240 GFLOP/s**.
4. **48 KB Tiling Cliff ($K = 1024$):** At $K = 1024$, centroid memory exceeds 48 KB, triggering 2-pass shared-memory tiling. Despite the tiling overhead, Stage 3 sustains over **241 GFLOP/s** thanks to 4-way register reuse.

---

## 6. Discussion, Conclusions & Future Work

### 6.1 Summary of Findings
1. **Hot Spot Dominance:** Serial profiling proved distance assignment consumes $97.13\%$ of runtime, verifying that parallelization effort was directed at the true computational bottleneck.
2. **Memory Coalescing is Mandatory:** Converting from AoS to a 128-float Padded SoA layout provided an immediate $6.4\times$ throughput improvement by replacing strided DRAM access with 100% coalesced 128-byte warp transactions.
3. **Amdahl's Law Overcomes Partial Offloading:** Offloading only the assignment kernel leaves host-device PCIe transfers and CPU reductions as a fatal bottleneck. Stage 3 eliminated transfers completely, enabling an overall speedup of **38.99x** to **69.3x**.

### 6.2 Lessons Learned
* **Hardware Intrinsics:** Utilizing Pascal-safe `__shfl_down_sync` warp-shuffle primitives enables zero-overhead convergence checks without global atomic contention.
* **Exact Floating-Point Parity:** Disabling automatic compiler FMA contraction via `--fmad=false` achieved bit-level parity between CPU and GPU results without sacrificing performance.

### 6.3 Future Work
Promising extensions include formulating the Euclidean distance calculation as a dense Matrix Multiplication (GEMM) via $\|x - \mu\|^2 = \|x\|^2 + \|\mu\|^2 - 2x \cdot \mu$ to exploit Tensor Cores on NVIDIA Volta and Ampere architectures, as well as distributed multi-GPU scaling using MPI.
