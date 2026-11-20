# Parallel K-Means Clustering: Pure C11 & CUDA Execution Plan

**Document Version:** 3.1 (Pure C11 Serial Baseline vs. Progressive CUDA C + Stateful Progress Tracking)  
**Target Course:** CSC 548 (Parallel Systems, Fall 2026)  
**Project Track:** Accelerate (Single-Person Project)  
**Target Workstation:** Intel Core i5-8400 (Serial CPU Reference) + NVIDIA GeForce GTX 1080 Ti (Pascal GP102, `sm_61`)  
**Key Deadlines:** In-Class Oral Presentation (Nov 23 or Nov 30, 2026; 10%) | Final Report + Code on Moodle (Dec 7, 2026, 11:59 PM ET; 20%)  
**Execution Standard:** Deterministic, self-contained engineering specification with live in-file progress tracking for autonomous execution.

---

## Agent Progress Tracking & Session Resume Protocol

This document is both the **authoritative technical specification** and the **live execution state file**. The autonomous agent MUST update this file in place as it works so progress persists cleanly across tool calls, context resets, or interruptions.

### Mandatory State-Update Rules
1. **Before Starting Any Task:** Change the task's status token from `NOT_STARTED` to `IN_PROGRESS` in **both** the Master Progress Dashboard below and the task's `**Status:**` line in Section D, and set `Current Active Task` in the dashboard header.
2. **During Task Execution:** Toggle each completed sub-step and verification item from `- [ ]` to `- [x]` immediately after verifying its command output.
3. **Upon Passing All Verification Gates:** Change the task status to `DONE`, update `Completed Tasks` count, record the concrete evidence/measured metric and UTC timestamp in the Master Progress Dashboard, and append one concise row to **Section F (Execution Log)**.
4. **On Failure or Blocker:** If a build, test, or environment dependency fails and cannot be resolved after 2 targeted retries, set status to `BLOCKED`, record the exact failing command and error message in **Section F.2 (Blocker & Deviation Register)**, and halt dependent tasks rather than skipping verification gates.
5. **Session Resume Rule:** Whenever starting or resuming work in a fresh context, read the **Master Progress Dashboard** and **Section F** first, verify the files from the most recent `DONE` task exist on disk, and resume from the first `IN_PROGRESS` or `NOT_STARTED` task.

### Master Progress Dashboard

- **Overall Progress:** `0 / 14` tasks completed (`0%`)
- **Current Phase:** `Phase 0`
- **Current Active Task:** `0.1`
- **Active Dataset Mode:** `PENDING` (`UCI_SUSY` or `SYNTHETIC_FALLBACK`)
- **Last Updated (UTC):** `PENDING`

| Task ID | Phase | Task Name | Status | Primary Artifact(s) | Evidence / Key Metric | Updated (UTC) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.1** | Phase 0 | Repo Layout, Hardware Probe & Pure-C Makefile | `NOT_STARTED` | `Makefile`, `include/*.h`, `results/hardware_specs.txt` | — | — |
| **0.2** | Phase 0 | Pure-C Data Ingestion, SplitMix64 Gen & Validation | `NOT_STARTED` | `include/input.h`, `src/csv_to_bin.c`, `scripts/prepare_susy.sh` | — | — |
| **0.3** | Phase 0 | Pure-C Verifier & Serial Baseline (`kmeans_seq`) | `NOT_STARTED` | `include/verify.h`, `src/kmeans_seq.c`, `bin/kmeans_seq` | — | — |
| **0.4** | Phase 0 | Serial Baseline Profiling (`gprof` / `perf`) | `NOT_STARTED` | `bin/kmeans_seq_prof`, `results/baseline_profile.txt` | — | — |
| **1.1** | Phase 1 | Pure-C CUDA Driver & Stage 1 Naive Kernel | `NOT_STARTED` | `include/kmeans_cuda.h`, `src/main_cuda.c`, `src/kmeans_cuda_naive.cu` | — | — |
| **1.2** | Phase 1 | Stage 2 Padded SoA & Tiled Shared-Memory Kernel | `NOT_STARTED` | `src/kmeans_cuda_shmem.cu`, `bin/kmeans_cuda_shmem` | — | — |
| **1.3** | Phase 1 | Stage 3 `float4` SoA + On-Device Reduction Kernel | `NOT_STARTED` | `src/kmeans_cuda_device.cu`, `bin/kmeans_cuda_device` | — | — |
| **2.1** | Phase 2 | Automated Correctness, Edge-Case & Determinism Suite | `NOT_STARTED` | `tests/run_tests.sh`, `results/test_suite_summary.txt` | — | — |
| **3.1** | Phase 3 | Unified CPU & GPU Profiler Audit | `NOT_STARTED` | `scripts/run_profiler.sh`, `results/cuda_profile.txt` | — | — |
| **3.2** | Phase 3 | Full Benchmark & Scaling Matrix Execution | `NOT_STARTED` | `scripts/run_benchmarks.sh`, `results/suite_*.csv` | — | — |
| **3.3** | Phase 3 | Dual PNG + Vector PDF Publication Figures | `NOT_STARTED` | `figures/fig_*.py`, `figures/*.png`, `figures/*.pdf` | — | — |
| **4.1** | Phase 4 | Reproducible `README.md` | `NOT_STARTED` | `README.md` | — | — |
| **4.2** | Phase 4 | ACM 2-Column Report & 4-Min Presentation Outline | `NOT_STARTED` | `report/report.md`, `report/report.tex`, `presentation/slides_outline.md` | — | — |
| **4.3** | Phase 4 | Final Pre-Submission Rubric Audit | `NOT_STARTED` | `results/final_audit.txt` | — | — |

---

## Section A: Project Instructions Traceability Matrix & Explicit Assumptions

### A.1 Requirements & Rubric Traceability Matrix

Every constraint from the CSC 548 Final Project guidelines, the approved proposal, the pure-C language constraint, the serial-vs-GPU scope decision, and the structural conventions of the `hw1` (`SpMV`) reference repository is mapped below to its implementing phase, artifacts, and verification gate.

| Req ID | Source Document | Requirement / Constraint | Mapped Plan Phase & Tasks | Deliverable / Artifact | Verification Gate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **REQ-01** | CSC 548 Module 1 & User Constraint | Strictly **pure C (C11)** for all CPU code, headers, and drivers (C-style `.cu` kernels with `extern "C"` linkage for GPU; Python only for offline plotting). Zero C++ STL, classes, templates, or `g++`. | Phase 0 (0.1–0.4), Phase 1 (1.1–1.3) | `src/*.c`, `src/*.cu`, `include/*.h` | `make all` compiles cleanly with `gcc -std=c11` and `nvcc` with zero warnings/errors. |
| **REQ-02** | CSC 548 Module 1 & 2 | Accelerate a real-world workload using an appropriate parallel model (CUDA on GPU) without over-scoping. | Phase 0 (0.2), Phase 1 (1.1–1.3) | `kmeans_cuda_naive`, `kmeans_cuda_shmem`, `kmeans_cuda_device` | Benchmarks run on real-world UCI SUSY ($N = 5{,}000{,}000, D = 18$) + built-in synthetic generator. |
| **REQ-03** | CSC 548 Module 4 & Proposal | Build and time a fair, compiler-optimized serial baseline first ($T_1$). | Phase 0 (0.3, 0.4) | `src/kmeans_seq.c`, `results/baseline_profile.txt` | Serial baseline (`-O3 -march=native`) passes determinism & ground-truth tests; $T_1$ logged. |
| **REQ-04** | CSC 548 Module 3 & 4 | Profile before optimizing; report what the profiler revealed and fixed ("never optimize blind"). | Phase 0 (0.4), Phase 3 (3.1) | `results/baseline_profile.txt` (`gprof`/`perf`), `results/cuda_profile.txt` (`nvprof`/`nsys`) | Quantifies serial hot spot ($> 90\%$ in assignment) and GPU global load efficiency / PCIe overhead. |
| **REQ-05** | Proposal & `Plan.md` Optimizations | Progressive CUDA kernels: 1 thread/point global baseline, shared-memory centroid broadcast + SoA coalescing, and on-GPU centroid reduction with `float4` loads, `__shfl_down_sync`, and pinned streams. | Phase 1 (1.1, 1.2, 1.3) | `src/kmeans_cuda_naive.cu`, `src/kmeans_cuda_shmem.cu`, `src/kmeans_cuda_device.cu` | Three distinct CUDA stages isolate global memory, shared-memory/SoA, and on-device optimization gains. |
| **REQ-06** | CSC 548 Module 4 & Proposal | Verify parallel outputs (assignments and centroids) match serial baseline ("Wrong answers, fast" pitfall). | Phase 2 (2.1) | `tests/run_tests.sh`, `results/*_check.txt` | Automated verifier asserts centroid relative $L_2$ error $\le 10^{-4}$, max abs error $\le 10^{-4}$ on well-conditioned tests, and label match $\ge 99.99\%$. |
| **REQ-07** | CSC 548 Module 4 | Avoid single data point: sweep dataset size $N$, cluster count $K$, dimensionality $D$, and CUDA thread block size. | Phase 3 (3.2), Section E | `scripts/run_benchmarks.sh`, `results/*.csv` | All 4 GPU benchmark suites populate CSV files with 5 timed trials + 1 warmup per configuration. |
| **REQ-08** | CSC 548 Module 4 | Show speedup ($S = T_{\text{serial}} / T_{\text{GPU}}$), scaling curves, GPU roofline vs. hardware ceiling, and Amdahl's Law bottleneck analysis. | Phase 3 (3.2, 3.3) | `figures/fig_*.py`, `figures/*.png`, `figures/*.pdf` | Generates scaling, breakdown, block-tuning, and roofline charts as 300 DPI PNG + vector PDF. |
| **REQ-09** | `hw1` (`SpMV`) Reference | Modular C repository layout (`Makefile`, `README.md`, `include/`, `src/`, `results/`, `figures/`) and clean C CLI/timing headers. | Section C, Phase 0 (0.1) | Repository tree, `Makefile`, `include/timer.h`, `include/cmdline.h` | Directory layout, pure C style, and verification log formatting match `hw1/SpMV`. |
| **REQ-10** | CSC 548 Module 3 | Final Report ($\ge 4$ pages excl. refs, ACM 2-column LaTeX) + Presentation (4 min + 30 s Q&A) + Code on Moodle. | Phase 4 (4.1, 4.2, 4.3) | `README.md`, `report/report.md`, `report/report.tex`, `presentation/slides_outline.md` | Rubric audit confirms all required sections, plots, tables, and reproducible build instructions exist. |

### A.2 Explicit Conservative Assumptions & Pure-C Policy

To eliminate guesswork during autonomous execution, the following rules are fixed. Every build parameter is overridable via `Makefile` variables or CLI flags without source edits:

1. **Strict Pure-C11 Mandate (Zero C++ & Zero OpenMP):**
   * All host programs (`csv_to_bin.c`, `kmeans_seq.c`, `main_cuda.c`) and all header files (`include/*.h`) MUST be strictly valid **ISO C11** (`-std=c11 -D_POSIX_C_SOURCE=200809L`) compiled with `gcc`.
   * **Single-Threaded Baseline Scope:** Do NOT implement an OpenMP CPU baseline; all speedups compare progressive CUDA GPU stages against the single-threaded, `-O3 -march=native` serial C11 baseline (`kmeans_seq`). (In the final report discussion, include an analytical comparison noting that even an ideal $6\times$ multicore CPU speedup on the 6-core i5-8400 remains far below the GTX 1080 Ti's throughput.)
   * **Forbidden Constructs:** Do NOT use `g++`, `.cpp`, `.hpp`, `.cuh`, C++ headers (`<vector>`, `<random>`, `<iostream>`, `<cmath>`, `<algorithm>`), namespaces, `std::`, classes, templates, `new`/`delete`, or C++ references (`&`).
   * **CUDA C Interoperability:** Write all `.cu` files strictly in the **C subset of CUDA** (using only `<stdio.h>`, `<stdlib.h>`, `<string.h>`, `<float.h>`, `<math.h>`, `<cuda_runtime.h>`, pointers, `typedef struct`, and C functions) and expose entry points via `#ifdef __cplusplus extern "C" { #endif` in `include/kmeans_cuda.h` so `src/main_cuda.c` links directly as pure C11.
2. **Operating System & Toolchain:** Linux x86_64 (Ubuntu 20.04/22.04/24.04, WSL2 with CUDA passthrough, or NCSU ARC Linux node). Compilers: `gcc` ($\ge 9.0$, C11) and `nvcc` (CUDA Toolkit 11.4–12.x). Python 3.8+ with `numpy`, `pandas`, and `matplotlib` used strictly for offline figure generation.
3. **GPU Architecture & Pascal (`sm_61`) Compatibility Rules:**
   * Default `CUDA_ARCH ?= sm_61` for the local **NVIDIA GTX 1080 Ti (Pascal GP102, Compute Capability 6.1)**, overridable via `make CUDA_ARCH=sm_80` for NCSU ARC Volta/Ampere nodes.
   * **Profiling Compatibility:** NVIDIA Nsight Compute (`ncu`) does **not** support Pascal (`sm_61`; requires Volta `sm_70+`). Therefore, `scripts/run_profiler.sh` MUST automatically probe in priority order: (1) `nvprof` (native Pascal hardware counters), (2) `nsys profile --stats=true` (Nsight Systems timeline/API/memcpy breakdown), (3) `ncu` only if GPU compute capability $\ge 7.0$, and (4) always emit the built-in `cudaEvent` timing and analytical bandwidth breakdown.
   * **Warp Primitive Compatibility:** Use `__shfl_down_sync(0xffffffffu, val, offset)` (fully supported on `sm_61` in CUDA 9+). Do **NOT** use Volta-only intrinsics such as `__match_any_sync`.
4. **Numerical Precision:** Single-precision IEEE-754 floating-point (`float`, 32-bit) is used across both CPU and GPU implementations. On Pascal GP102, FP32 throughput is $11{,}340\text{ GFLOP/s}$ whereas FP64 runs at $1/32$ rate ($354\text{ GFLOP/s}$). Compilation **forbids** `-ffast-math` so C11 `<math.h>` `isfinite()` checks for `NaN`/`Inf` remain valid.
5. **Dataset Strategy (Primary Real-World + Optional Proposal Dataset + Synthetic):**
   * **Primary Real-World Benchmark:** UCI SUSY ($N = 5{,}000{,}000$, $D = 18$ numeric features after skipping column 0 class label, $\approx 360\text{ MB}$ binary FP32).
   * **Optional Proposal Dataset Support:** `src/csv_to_bin.c` supports `--delim ';' --skip-cols 2 --skip-missing` so it can also ingest the *UCI Individual Household Electric Power Consumption* dataset ($N \approx 2{,}049{,}280$ valid rows, $D = 7$) without separate parser code.
   * **Offline Resilience:** If `data/susy.bin` is unavailable during automated testing, all correctness checks and sweeps run seamlessly via the built-in pure-C deterministic Gaussian mixture generator (`--synthetic`), and the agent records `Active Dataset Mode: SYNTHETIC_FALLBACK` in the Master Progress Dashboard.

---

## Section B: System Architecture, Data Model & Parallelization Strategy

### B.1 Hardware Roofline Parameters

| Metric | Intel Core i5-8400 (1 Core Serial Baseline) | NVIDIA GTX 1080 Ti (Pascal GP102) | Architectural Ratio (GPU / 1C CPU) |
| :--- | :--- | :--- | :--- |
| Execution Units | 1 physical core (256-bit AVX2 SIMD) | 28 SMs / 3,584 CUDA cores | $3{,}584$ SIMT lanes |
| Clock Speed | $4.0\text{ GHz}$ single-core turbo | $1.582\text{ GHz}$ boost | $0.40\times$ |
| Peak FP32 Compute | $128.0\text{ GFLOP/s}$ (AVX2 FMA) | $11{,}340\text{ GFLOP/s}$ | $88.6\times$ |
| Peak Memory Bandwidth | $41.6\text{ GB/s}$ (DDR4-2666 2-ch) | $484.0\text{ GB/s}$ (GDDR5X 352-bit) | $11.6\times$ |
| On-Chip Fast Memory | $32\text{ KB}$ L1d, $256\text{ KB}$ L2, $9\text{ MB}$ L3 | $48\text{ KB}$ shared mem/block ($96\text{ KB}$/SM) | Software-managed broadcast |
| Host-Device Interconnect | N/A | PCIe 3.0 x16 ($\approx 12.5\text{ GB/s}$ pinned) | Major bottleneck if inside loop |
| Roofline Ridge Point | $3.08\text{ FLOP/byte}$ | $23.43\text{ FLOP/byte}$ | Governs memory vs. compute regime |

### B.2 Algorithmic Decomposition & Arithmetic Intensity Analysis

Lloyd's K-Means algorithm partitions $N$ points $X = \{x_0, \dots, x_{N-1}\} \subset \mathbb{R}^D$ into $K$ clusters with centroids $M = \{\mu_0, \dots, \mu_{K-1}\} \subset \mathbb{R}^D$ over $I$ iterations:

1. **Expectation / Assignment Step (Embarrassingly Data-Parallel):**
   For each point $i \in \{0, \dots, N-1\}$ independently, find the nearest centroid using squared Euclidean distance:
   $$a_i^{(t)} = \arg\min_{c \in \{0, \dots, K-1\}} \sum_{d=0}^{D-1} \left( x_{i,d} - \mu_{c,d}^{(t)} \right)^2$$
   * **Exact Tie-Breaking Rule:** Initialize `best_c = 0`, `best_dist = FLT_MAX`. Update only on strict inequality `if (dist < best_dist)`. Exact ties deterministically choose the lowest centroid index $c$.
   * **FLOP Count per Iteration:** Each $(i, c, d)$ step executes 1 subtraction, 1 multiplication, and 1 addition ($3\text{ FLOPs}$), yielding $3 N K D$ FLOPs for assignment.

2. **Maximization / Centroid Update Step (Parallel Reduction):**
   For each cluster $c \in \{0, \dots, K-1\}$:
   $$n_c^{(t)} = \sum_{i=0}^{N-1} \mathbb{I}\!\left[a_i^{(t)} = c\right], \qquad S_{c,d}^{(t)} = \sum_{i : a_i^{(t)} = c} x_{i,d}$$
   $$\mu_{c,d}^{(t+1)} = \begin{cases} S_{c,d}^{(t)} / n_c^{(t)} & \text{if } n_c^{(t)} > 0 \\ \mu_{c,d}^{(t)} & \text{if } n_c^{(t)} = 0 \text{ (deterministic empty-cluster fallback)} \end{cases}$$
   * **Total Standardized FLOP Formula:**
     $$F_{\text{iter}} = 3 N K D + N D + K D = N D (3K + 1) + K D$$

3. **Arithmetic Intensity & Regime Transition:**
   When centroids are cached in GPU shared memory (Stages 2 and 3), the assignment kernel reads each point $x_i$ ($4D$ bytes) once from GDDR5X and writes 1 integer label ($4$ bytes):
   $$\text{AI}_{\text{assign}} = \frac{3 N K D}{4 N D + 4 N} = \frac{3 K D}{4 (D + 1)} \text{ FLOP/byte}$$
   * For UCI SUSY ($D = 18$), $\text{AI}_{\text{assign}} \approx 0.7105 \cdot K$.
   * For $K \in \{8, 16\}$ ($\text{AI} = 5.7, 11.4\text{ FLOP/byte}$), the kernel is **memory-bandwidth bound**.
   * Around $K \approx 33$ ($\text{AI} \approx 23.4\text{ FLOP/byte}$), the kernel crosses the **GTX 1080 Ti ridge point**.
   * For $K \in \{64, 128, 256, 512, 1024\}$, the kernel enters the **compute / shared-memory-throughput bound** regime, allowing GPU speedup over the serial CPU to far exceed the $11.6\times$ raw memory bandwidth ratio.

### B.3 Memory Layout Strategy (AoS vs. 128-Float Padded SoA for `float4` Vectorization)

Two contiguous 1D flat `float *` buffers are explicitly defined:

1. **Row-Major Array-of-Structures (AoS) — Serial CPU (`IMPL-0`) & GPU Naive (`IMPL-1`):**
   * Element $x_{i,d}$ stored at `points_aos[(size_t)i * D + d]`.
   * **Why on CPU:** Streaming across $D$ contiguous floats maximizes L1 cache-line spatial locality ($64\text{ bytes} = 16$ floats) and enables `gcc -O3 -march=native` AVX2 auto-vectorization.
   * **Why on GPU Stage 1 (Naive):** Adjacent threads in a 32-thread warp access `points_aos[(size_t)i * D + d]` with a stride of $4D$ bytes ($72$ bytes for SUSY), intentionally demonstrating **uncoalesced global memory transactions** for profiler contrast against Stage 2.

2. **128-Float Padded Structure-of-Arrays (SoA) — GPU Shared-Memory (`IMPL-2`) & GPU Optimized Device (`IMPL-3`):**
   * Define padded stride $N_{\text{pad}} = \lceil N / 128 \rceil \times 128$ (multiple of 128 floats = 512 bytes = 32 `float4` vectors).
   * Element $x_{i,d}$ stored at `points_soa[(size_t)d * N_pad + i]`, with tail elements $i \in [N, N_{\text{pad}} - 1]$ zero-filled.
   * **Why $N_{\text{pad}}$ is a multiple of 128:**
     * In **Stage 2** (1 `float` per thread), every row `d * N_pad` is 512-byte aligned, so warp lanes $0 \dots 31$ load 32 contiguous floats in a single 128-byte coalesced transaction.
     * In **Stage 3** (1 `float4` = 4 points per thread via thread coarsening), every row pointer `d_points_soa + (size_t)d * N_pad` is guaranteed 16-byte aligned for `LDG.E.128` vectorized `float4` loads. Warp lanes $0 \dots 31$ load 32 contiguous `float4`s (512 bytes = 4 full 128-byte cache lines, 100% coalesced), while reusing each shared-memory centroid broadcast across 4 points in registers (cutting shared-memory read traffic by $4\times$).
   * **Layout Cost Accounting:** Host transpose `convert_aos_to_soa()` is timed in `setup_ms` and kept distinct from per-iteration `total_compute_ms`, allowing both **compute-only speedup** and **end-to-end speedup** (`setup_ms + total_compute_ms`) to be reported honestly.

### B.4 Four-Implementation Progression & Parallel Systems Concepts Demonstrated

| Impl ID | Binary Name | Language / Compiler | Memory Layout | Assignment Execution | Centroid Update & Transfers | Parallel Systems Concept Demonstrated |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IMPL-0** | `kmeans_seq` | Pure C11 (`gcc -O3 -march=native`) | AoS (Host) | Single-threaded CPU loop | Single-threaded CPU sums | Fair optimized serial baseline ($T_1$); `gprof`/`perf` hot-spot proof ($> 90\%$ assignment). |
| **IMPL-1** | `kmeans_cuda_naive` | C11 driver + C-style CUDA | AoS (Device) | 1 thread/point; points & centroids from global memory | Per-iteration D2H `assignments` ($4N$ B) $\to$ CPU update $\to$ H2D `centroids` ($4KD$ B) | Naive GPU offload; **uncoalesced DRAM reads**; **PCIe communication bottleneck** (Amdahl's Law). |
| **IMPL-2** | `kmeans_cuda_shmem` | C11 driver + C-style CUDA | Padded SoA (Device) | 1 thread/point; coalesced SoA reads + **tiled shared-memory centroid broadcast** | Per-iteration D2H `assignments` $\to$ CPU update $\to$ H2D `centroids` | **Global memory coalescing**; **shared-memory broadcast** (0 bank conflicts); **48 KB tiling cliff**. |
| **IMPL-3** | `kmeans_cuda_device` | C11 driver + C-style CUDA | Padded SoA (Device) | **`float4` 4-point thread-coarsened** coalesced SoA reads + tiled shared-memory centroids | **100% on-device:** two-tier shared-memory block reduction + global flush + `normalize_kernel` with **`__shfl_down_sync` warp reduction**; pinned `cudaHostAlloc` + `cudaStream_t`. **Zero per-iteration PCIe transfers.** | Eliminating host-device sync barrier; **vectorized `LDG.E.128` + register reuse**; **hierarchical atomic reduction**; warp-shuffle reductions; pinned async DMA. |

### B.5 Exact Engineering Rules for GPU Hazards & Pure-C Determinism

1. **GPU Shared-Memory Sizing & Tiling Cliff (Pascal 48 KB = 12,288 `float`s):**
   * Define `#define SHMEM_MAX_FLOATS 12288` (49,152 bytes).
   * **Assignment Tiling (Stages 2 & 3):**
     * Compute `int k_tile = SHMEM_MAX_FLOATS / D; if (k_tile > K) k_tile = K;`
     * Loop `for (int c_base = 0; c_base < K; c_base += k_tile)`:
       * `int c_count = (K - c_base < k_tile) ? (K - c_base) : k_tile;`
       * Cooperatively load `c_count * D` floats into `extern __shared__ float s_centroids[];`, call `__syncthreads()`, evaluate distances against `s_centroids[c_local * D + d]` (broadcast across all 32 warp lanes with **0 bank conflicts**), and call `__syncthreads()` before loading the next tile.
     * For UCI SUSY ($D = 18$), `k_tile = 12288 / 18 = 682`. Thus $K \le 512$ fits in 1 tile, while $K = 1024$ triggers 2-pass tiling (`682 + 342`), cleanly exposing the 48 KB hardware cliff in Suite D.

2. **Stage 3 `float4` Vectorized Assignment Kernel (`assign_opt_float4_kernel`):**
   * Launch over `num_vec = N_pad / 4` threads (`grid = (num_vec + block_size - 1) / block_size`).
   * Thread `tid < num_vec` handles 4 contiguous points `base_i = tid * 4 + {0, 1, 2, 3}`.
   * Maintain 4 register accumulators `best_dist0..3 = FLT_MAX` and `best_c0..3 = 0`.
   * Inside each tile `c_local`, accumulate `dist0..3 = 0.0f` over `d = 0 .. D-1`:
     * Read `float4 x = ((const float4 *)(d_points_soa + (size_t)d * N_pad))[tid];`
     * Broadcast `float mu = s_centroids[(size_t)c_local * D + d];`
     * Compute `float diff0 = x.x - mu; dist0 += diff0 * diff0;` (and identically for `x.y`, `x.z`, `x.w`).
   * Guard global writes against padding indices: `if (base_i + 0 < N) d_assignments[base_i + 0] = best_c0;` (and `+1, +2, +3`).
   * Because each point computes the exact same floating-point operations in order `d = 0 .. D-1`, assignment outputs remain numerically identical to Stage 2.

3. **Stage 3 Two-Tier On-Device Centroid Reduction & Warp-Shuffle Normalization:**
   * Keep `assign_opt_float4_kernel`, `accumulate_kernel`, and `normalize_centroids_kernel` separate so assignment retains maximum register and shared-memory occupancy.
   * **Path A — Block-Private Shared-Memory Reduction** when `(size_t)K * D + (size_t)K <= 6144` floats ($\le 24\text{ KB}$, allowing $\ge 2\text{--}4$ active blocks per SM on Pascal's 96 KB SM SRAM):
     * Shared memory contains `float *s_sums = s_buf;` (`K * D` floats) and `int *s_counts = (int *)(s_buf + (size_t)K * D);` (`K` ints).
     * Cooperatively zero `s_sums` and `s_counts`, call `__syncthreads()`, accumulate thread `i < N` into `s_sums` and `s_counts` via shared-memory `atomicAdd`, call `__syncthreads()`, and cooperatively flush non-zero entries to global `d_new_centroids` and `d_counts` via global `atomicAdd`.
   * **Path B — Direct Global Atomic Fallback** when `(size_t)K * D + (size_t)K > 6144` floats:
     * Each thread `i < N` directly executes `atomicAdd(&d_counts[c], 1)` and `atomicAdd(&d_new_centroids[(size_t)c * D + d], d_points_soa[(size_t)d * N_pad + i])`.
   * **Warp-Shuffle Normalization (`normalize_centroids_kernel`):**
     * Define Pascal-safe helper:
       ```c
       __device__ __forceinline__ float warp_reduce_sum_f32(float val) {
           for (int offset = 16; offset > 0; offset >>= 1) {
               val += __shfl_down_sync(0xffffffffu, val, offset);
           }
           return val;
       }
       ```
     * Launch 1D grid covering `K * D` elements (`threads = 256`, `blocks = (K * D + 255) / 256`).
     * Each thread computes `float local_shift = 0.0f;` for `idx < K * D`: if `d_counts[c] > 0`, updates `d_centroids[idx] = new_val` and sets `local_shift = diff * diff;`; else retains `d_centroids[idx]`.
     * All threads in the warp participate in `float warp_shift = warp_reduce_sum_f32(local_shift);`, and lane 0 (`(threadIdx.x & 31) == 0`) performs a single `atomicAdd(d_shift_sq, warp_shift)` when `early_stop` is enabled—cutting global atomic writes for convergence checking by $32\times$.
   * **Pinned Memory & CUDA Stream (`cudaHostAlloc` + `cudaStream_t`):**
     * Create an explicit non-default stream `cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));`.
     * Allocate host SoA staging buffer and pinned output buffers with `cudaHostAlloc(..., cudaHostAllocDefault)` and use `cudaMemcpyAsync` / `cudaMemsetAsync` on `stream`, synchronizing cleanly at measurement boundaries.

4. **Pure-C Deterministic PRNG & Input Validation Rules (No C++ `<random>`):**
   * Implement a self-contained 64-bit **SplitMix64** PRNG in `include/input.h` for cross-compiler bit-exact determinism:
     ```c
     typedef struct {
         uint64_t state;
     } RngState;

     static inline uint64_t rng_next_u64(RngState *rng) {
         uint64_t z = (rng->state += 0x9e3779b97f4a7c15ULL);
         z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
         z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
         return z ^ (z >> 31);
     }

     /* Uniform double strictly in (0, 1] for safe log() in Box-Muller */
     static inline double rng_uniform_open0(RngState *rng) {
         return ((double)(rng_next_u64(rng) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
     }
     ```
   * **Synthetic Generator (`generate_synthetic_points`):** Seed `RngState rng = { (uint64_t)seed }`. Generate $K$ ground-truth cluster centers uniformly in $[-10.0, 10.0]^D$. For each point $i \in [0, N)$, set `g = i % K` and add standard normal noise $\mathcal{N}(0, 1^2)$ via Box-Muller (`sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2)`), defining `#ifndef M_PI` `#define M_PI 3.14159265358979323846` `#endif` for strict `-std=c11`.
   * **Deterministic Centroid Initialization (`init_centroids_deterministic`):** Seed `RngState rng = { (uint64_t)seed + 1337ULL }`. Perform a partial Fisher-Yates shuffle on `idx[0..N-1]` for `c = 0 .. K-1` and copy point `idx[c]` into centroid `c`.
   * **Mandatory Input Validation (`validate_dataset`):**
     * Reject `N <= 0`, `D <= 0`, `K <= 0`, `max_iters <= 0`, `block_size <= 0`, `block_size > 1024`, or `block_size % 32 != 0` with exit code `1`.
     * Reject `N < K` or `D > 12288` with exit code `3`.
     * Scan all `N * D` floats using C11 `<math.h>` `isfinite(val)`; if any `NaN` or `Inf` is found, print `(point_idx, dim_idx)` to `stderr` and exit with code `3`.

---

## Section C: Repository Layout & CLI / API Contract

### C.1 Exact Directory Tree

```text
parallel-kmeans/
├── Makefile
├── README.md
├── .gitignore
├── include/
│   ├── alloc.h                 # 64-byte aligned malloc/free helpers (C11)
│   ├── cmdline.h               # Deterministic C11 CLI parser
│   ├── config.h                # Hardware constants, tolerances, exit codes
│   ├── input.h                 # Binary .bin loader, SplitMix64 Box-Muller gen, AoS->SoA, validation
│   ├── kmeans_cuda.h           # Pure C / extern "C" interface between main_cuda.c and *.cu
│   ├── kmeans_types.h          # typedef structs: KMeansConfig, KMeansMetrics, VerifyResult
│   ├── timer.h                 # CLOCK_MONOTONIC wall_seconds() wrapper
│   └── verify.h                # Serial CPU reference, L2/max-abs/label verification, ref I/O, CSV writer
├── src/
│   ├── csv_to_bin.c            # Pure C11 UCI SUSY & UCI Power CSV -> binary .bin converter
│   ├── kmeans_seq.c            # IMPL-0: Pure C11 serial baseline binary
│   ├── main_cuda.c             # Shared Pure C11 CLI driver + verification harness for all CUDA stages
│   ├── kmeans_cuda_naive.cu    # IMPL-1: Stage 1 C-style CUDA AoS global memory + host update
│   ├── kmeans_cuda_shmem.cu    # IMPL-2: Stage 2 C-style CUDA Padded SoA + tiled shared-memory centroids
│   └── kmeans_cuda_device.cu   # IMPL-3: Stage 3 C-style CUDA float4 SoA + shared mem + on-device reduction + warp shuffle
├── scripts/
│   ├── prepare_susy.sh         # Downloads & converts UCI SUSY CSV -> data/susy.bin
│   ├── run_profiler.sh         # Profiles serial CPU (gprof/perf) & GPU (nvprof/nsys/ncu/events)
│   └── run_benchmarks.sh       # Executes Suites B, C, D, E -> results/*.csv
├── tests/
│   └── run_tests.sh            # Automated correctness, determinism, and edge-case test suite
├── figures/
│   ├── fig_scaling.py          # Generates speedup_vs_n.{png,pdf}, gpu_saturation.{png,pdf}
│   ├── fig_k_and_block.py      # Generates scaling_vs_k.{png,pdf}, block_size_tuning.{png,pdf}, dim_scaling.{png,pdf}
│   ├── fig_breakdown.py        # Generates time_breakdown.{png,pdf} (stacked assign/update/xfer)
│   └── fig_roofline.py         # Generates roofline.{png,pdf} (GTX 1080 Ti + i5-8400 ceilings)
├── data/                       # Ignored by git; holds susy.bin & test fixtures
├── results/                    # Hardware specs, check logs, profiler dumps, benchmark CSVs
├── report/
│   ├── report.md               # Structured Markdown draft
│   └── report.tex              # ACM 2-column (sigconf) LaTeX final report
└── presentation/
    └── slides_outline.md       # 4-minute presentation deck script & visual mapping
```

### C.2 Build System Contract (`Makefile`)

* **Configurable Variables (Strict Pure C11 + `nvcc`):**
  * `CC ?= gcc`
  * `NVCC ?= nvcc`
  * `CUDA_ARCH ?= sm_61`
  * `CFLAGS := -O3 -std=c11 -D_POSIX_C_SOURCE=200809L -march=native -Wall -Wextra -Iinclude` (No `-ffast-math`).
  * `LDFLAGS := -lm`
  * `NVCCFLAGS := -O3 -arch=$(CUDA_ARCH) -lineinfo -Iinclude -Xcompiler "-Wall -Wextra"`
* **Compilation & Linking Rules:**
  * `bin/csv_to_bin`: `$(CC) $(CFLAGS) src/csv_to_bin.c -o $@ $(LDFLAGS)`
  * `bin/kmeans_seq`: `$(CC) $(CFLAGS) src/kmeans_seq.c -o $@ $(LDFLAGS)`
  * `bin/kmeans_seq_prof`: `$(CC) $(CFLAGS) -pg -fno-inline src/kmeans_seq.c -o $@ $(LDFLAGS)` (dedicated `gprof` profiling target)
  * `obj/main_cuda.o`: `$(CC) $(CFLAGS) -c src/main_cuda.c -o $@` (compiled as **pure C11**)
  * `obj/kmeans_cuda_%.o`: `$(NVCC) $(NVCCFLAGS) -c src/kmeans_cuda_$*.cu -o $@`
  * `bin/kmeans_cuda_%`: `$(NVCC) -arch=$(CUDA_ARCH) obj/main_cuda.o obj/kmeans_cuda_$*.o -o $@ $(LDFLAGS)`
* **Mandatory Targets:** `all` (`csv_to_bin`, `kmeans_seq`, `kmeans_cuda_naive`, `kmeans_cuda_shmem`, `kmeans_cuda_device`), `cpu`, `prof` (`bin/kmeans_seq_prof`), `cuda`, `test`, `clean` (`clean` removes `bin/`, `obj/`, and `gmon.out` only; never deletes `data/` or `results/`).

### C.3 Binary Data File Format (`.bin`)

Every binary dataset file written by `bin/csv_to_bin` and read by `load_binary_dataset()` must obey this exact little-endian layout:

* Bytes `0..7`: `uint64_t num_points` ($N$)
* Bytes `8..15`: `uint64_t num_dims` ($D$)
* Bytes `16..(16 + 4*N*D - 1)`: `float data[N * D]` stored in row-major AoS order.

### C.4 Unified CLI Contract (All 4 K-Means Binaries)

Every K-Means binary (`kmeans_seq`, `kmeans_cuda_naive`, `kmeans_cuda_shmem`, `kmeans_cuda_device`) MUST accept the identical flag set:

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--file <path>` | `const char *` | `""` | Path to `.bin` dataset file (mutually exclusive with `--synthetic`). |
| `--synthetic` | `bool` | `false` | Generate synthetic Gaussian clusters in memory using `--n`, `--d`, `--k`, `--seed`. |
| `--n <int>` | `int` | `0` | Number of points $N$. If `--file` is set and `--n > 0`, slices first $N$ rows; if `0`, loads all rows. |
| `--d <int>` | `int` | `18` | Dimensionality $D$ (used only with `--synthetic`; overridden by header if `--file` is used). |
| `--k <int>` | `int` | `32` | Number of clusters $K$. |
| `--iters <int>` | `int` | `20` | Maximum number of Lloyd iterations $I$. |
| `--fixed-iters` | `bool` | `true` | Run exact `--iters` iterations (disables early convergence break for fair benchmarking). |
| `--early-stop` | `bool` | `false` | Enable early stopping when total squared centroid shift $\le \text{tol}^2$. |
| `--tol <float>` | `float` | `1e-5f` | Convergence tolerance for `--early-stop`. |
| `--seed <uint>` | `uint32_t` | `42` | Deterministic seed for SplitMix64 synthetic generation and initial centroid selection. |
| `--block-size <int>` | `int` | `256` | CUDA thread block size (multiple of 32, $\in [32, 1024]$; logged as `0` on `kmeans_seq`). |
| `--warmup <int>` | `int` | `1` | Number of untimed warmup runs executed before timed trials. |
| `--trials <int>` | `int` | `1` | Number of timed benchmark trials; reports median metrics across trials (sorted with C `qsort`). |
| `--verify` | `bool` | `false` | Run in-process serial CPU reference on identical input & centroids and assert parity. |
| `--dump-ref <prefix>` | `const char *` | `""` | Write `<prefix>_centroids.bin` and `<prefix>_labels.bin` after execution. |
| `--check-ref <prefix>` | `const char *` | `""` | Load cached reference files and assert parity without re-running serial CPU. |
| `--csv <path>` | `const char *` | `""` | Append one formatted CSV metric row to `<path>` (creates header if file does not exist). |
| `--csv-header` | `bool` | `false` | Print CSV header line to `stdout` and exit `0`. |

### C.5 Standardized Exit Codes

| Exit Code | Constant Name | Meaning |
| :--- | :--- | :--- |
| `0` | `EXIT_OK` | Success (and `--verify` / `--check-ref` passed if requested). |
| `1` | `EXIT_CLI_ERR` | Invalid CLI arguments or out-of-range parameters (`block_size`, `iters`, missing input mode). |
| `2` | `EXIT_IO_ERR` | File I/O failure (missing `.bin` file, truncated binary header, unreadable reference file). |
| `3` | `EXIT_DATA_ERR` | Data validation error (`N < K`, `D <= 0`, `D > 12288`, or `NaN`/`Inf` found in input coordinates). |
| `4` | `EXIT_CUDA_ERR` | CUDA runtime error (checked via `CUDA_CHECK(call)` macro printing file, line, error string). |
| `5` | `EXIT_VERIFY_ERR` | Verification failure (centroid relative $L_2$ error $> 10^{-4}$ or label agreement $< 99.99\%$). |

### C.6 Standardized CSV Output Schema (21 Columns)

All benchmark CSV files must use this exact header:

```text
impl,dataset,N,D,K,iters_run,block_size,trial_count,io_ms,setup_ms,assign_ms,update_ms,xfer_ms,total_compute_ms,ms_per_iter,gflops,gbytes_s,verify_status,centroid_rel_l2,centroid_max_abs,label_match_pct
```

* **Standardized `gbytes_s` Formula (Effective Algorithmic Bandwidth):**
  $$\text{Bytes}_{\text{iter}} = 4 \cdot (N \cdot D + N + K \cdot D), \qquad \text{GB/s} = \frac{\text{Bytes}_{\text{iter}} \cdot I}{\text{total\_compute\_ms} \times 10^6}$$
* **Standardized `gflops` Formula:**
  $$\text{FLOPs}_{\text{iter}} = N \cdot D \cdot (3K + 1) + K \cdot D, \qquad \text{GFLOP/s} = \frac{\text{FLOPs}_{\text{iter}} \cdot I}{\text{total\_compute\_ms} \times 10^6}$$

---

## Section D: Phased Execution Plan for Execution Agent

Execute each phase and atomic task strictly in order. Toggle each checkbox (`- [ ]` $\to$ `- [x]`) and update `**Status:**` + the **Master Progress Dashboard** as each gate passes.

### Phase 0: Environment, Pure-C Infrastructure & Sequential Baseline Reference

#### Task 0.1: Initialize Repository Layout, Hardware Probe & Pure-C Build System
**Status:** `NOT_STARTED` | **Completed (UTC):** `—`

1. **Objective & Requirement Mapping:** Satisfies **REQ-01**, **REQ-09**. Establishes clean directory tree, pure-C11 headers, and `Makefile` modeled after `hw1/SpMV`.
2. **Exact Technical Decisions:**
   * Create directories: `include/`, `src/`, `scripts/`, `tests/`, `figures/`, `data/`, `results/`, `report/`, `presentation/`, `bin/`, `obj/`.
   * Use standard C include guards (`#ifndef KMEANS_*_H` / `#define KMEANS_*_H` / `#endif`) in every header, plus `#ifdef __cplusplus extern "C" { #endif` guards in `config.h`, `kmeans_types.h`, `timer.h`, `alloc.h`, `input.h`, and `kmeans_cuda.h`.
   * `include/config.h`: Define `EXIT_OK 0`, `EXIT_CLI_ERR 1`, `EXIT_IO_ERR 2`, `EXIT_DATA_ERR 3`, `EXIT_CUDA_ERR 4`, `EXIT_VERIFY_ERR 5`, `SHMEM_MAX_FLOATS 12288`, `ACCUM_SHMEM_MAX_FLOATS 6144`, `VERIFY_REL_L2_TOL 1e-4`, `VERIFY_LABEL_MATCH_MIN 99.99`, and `M_PI` fallback.
   * `include/timer.h`: Provide `static inline double wall_seconds(void)` using `clock_gettime(CLOCK_MONOTONIC, &ts)`.
   * `include/alloc.h`: Provide `static inline void *aligned_alloc_64(size_t bytes)` via `posix_memalign(&ptr, 64, bytes)` (allocating 64 bytes if `bytes == 0`) and `static inline void aligned_free_64(void *ptr)`.
   * `include/kmeans_types.h`: Define `KMeansConfig`, `KMeansMetrics`, and `VerifyResult` (including `centroid_rel_l2`, `centroid_max_abs`, `label_match_pct`, `passed`) using fixed-size `char` paths (`512` bytes) and `<stdbool.h>` / `<stdint.h>`.
   * `include/cmdline.h`: Implement `static inline int parse_args(int argc, char **argv, KMeansConfig *cfg)` validating all flags from Section C.4 using `strcmp`, `strtol`, `strtoul`, and `strtof`.
3. **Implementation Steps:**
   - [ ] Create all directories and `.gitignore` (ignoring `bin/`, `obj/`, `gmon.out`, `data/*.csv`, `data/*.bin`, `data/*.gz`, `data/*.txt`, `.venv/`).
   - [ ] Write `include/config.h`, `include/timer.h`, `include/alloc.h`, `include/kmeans_types.h`, and `include/cmdline.h`.
   - [ ] Write `Makefile` matching Section C.2 (`gcc -std=c11` for all `.c` files, `nvcc` for `.cu` files, zero OpenMP flags).
   - [ ] Probe and log host hardware to `results/hardware_specs.txt` capturing `uname -a`, `lscpu`, `free -h`, `gcc --version`, `nvcc --version`, and `nvidia-smi -q`.
4. **Verification & Acceptance Criteria:**
   - [ ] `ls -la include/ results/hardware_specs.txt Makefile` succeeds.
   - [ ] Zero `.cpp`, `.hpp`, or `.cuh` files exist in the repository.
   - [ ] `results/hardware_specs.txt` records CPU model, `gcc` version, `nvcc` version, and GPU model.

---

#### Task 0.2: Implement Pure-C Data Ingestion, SplitMix64 Synthetic Generator, Layout Conversion & Validation
**Status:** `NOT_STARTED` | **Completed (UTC):** `—`

1. **Objective & Requirement Mapping:** Satisfies **REQ-01**, **REQ-02**, **REQ-06**. Guarantees bit-identical input loading, 128-float padded AoS-to-SoA transforms, deterministic centroid seeding, and strict edge-case validation in pure C11.
2. **Exact Technical Decisions:**
   * `include/input.h` exact pure-C signatures:
     * `static inline int load_binary_dataset(const char *path, int n_limit, float **out_aos, int *out_N, int *out_D);`
     * `static inline int generate_synthetic_points(int N, int D, int K, uint32_t seed, float **out_aos);` (SplitMix64 + Box-Muller from Section B.5)
     * `static inline int validate_dataset(const float *aos, int N, int D, int K);` (checks `N < K`, `D <= 0`, `D > 12288`, and C11 `isfinite(aos[idx])` on all `N * D` floats)
     * `static inline void init_centroids_deterministic(const float *aos, int N, int D, int K, uint32_t seed, float *out_centroids);` (partial Fisher-Yates via SplitMix64)
     * `static inline int compute_padded_n(int N);` returns `((N + 127) / 128) * 128` (128-float alignment for both coalesced `float` and vectorized `float4` SoA access)
     * `static inline void convert_aos_to_soa(const float *aos, int N, int D, float *soa, int N_pad);` (zero-fills padding slots `N .. N_pad - 1`)
   * `src/csv_to_bin.c`: Pure C11 program supporting:
     * `bin/csv_to_bin --in <csv> --out <bin> [--delim <char>] [--skip-cols <int>] [--skip-header] [--skip-missing] [--max-rows <int>]`
     * Default `--delim ','`, `--skip-cols 1` (for UCI SUSY). Setting `--delim ';' --skip-cols 2 --skip-header --skip-missing` cleanly parses the UCI Individual Household Electric Power Consumption dataset (dropping rows containing `?` or empty tokens).
     * Reads lines via `fgets` (65,536-byte buffer), parses with `strtof`, verifies `isfinite()`, writes a 16-byte placeholder header, streams buffered FP32 rows, then `fseek(fp, 0, SEEK_SET)` to write final `uint64_t num_points` and `uint64_t num_dims`.
   * `scripts/prepare_susy.sh`: Downloads `https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz`, streams `gzip -dc` into `bin/csv_to_bin --in - --out data/susy.bin --skip-cols 1` (supporting `-` for `stdin`), and verifies `stat -c%s data/susy.bin` equals `360000016` bytes.
3. **Implementation Steps:**
   - [ ] Implement `include/input.h` and `src/csv_to_bin.c`.
   - [ ] Implement `scripts/prepare_susy.sh` and `chmod +x scripts/prepare_susy.sh`.
   - [ ] Compile `make bin/csv_to_bin`.
   - [ ] Test `bin/csv_to_bin` on both a 10-row comma CSV (`--skip-cols 1`) and a 5-row semicolon CSV containing 1 `?` row (`--delim ';' --skip-cols 2 --skip-header --skip-missing`).
   - [ ] If network access and disk space ($\ge 4\text{ GB}$) are available, run `bash scripts/prepare_susy.sh` and update `Active Dataset Mode` in the Master Progress Dashboard.
4. **Verification & Acceptance Criteria:**
   - [ ] `make bin/csv_to_bin` compiles cleanly with `gcc -std=c11 -Wall -Wextra` and zero warnings.
   - [ ] Round-trip test on 10-row, 5-column CSV produces exactly `16 + 10 * 4 * 4 = 176` bytes; semicolon `?`-handling test drops the malformed row cleanly.
   - [ ] If SUSY was downloaded, `stat -c%s data/susy.bin` outputs `360000016` (else `SYNTHETIC_FALLBACK` is documented in Section F).

---

#### Task 0.3: Implement Pure-C Verification Harness & Deterministic Serial Baseline (`kmeans_seq.c`)
**Status:** `NOT_STARTED` | **Completed (UTC):** `—`

1. **Objective & Requirement Mapping:** Satisfies **REQ-01**, **REQ-03**, **REQ-06**. Builds the fair, compiler-optimized ground-truth sequential Lloyd's K-Means engine (`IMPL-0`) and reusable pure-C verification module.
2. **Exact Technical Decisions:**
   * `include/verify.h` exact pure-C signatures:
     * `static inline void kmeans_cpu_assign(const float *points_aos, const float *centroids, int *assignments, int N, int D, int K);` (separate non-inlined function in `kmeans_seq.c` so `gprof` cleanly separates assignment vs. update)
     * `static inline void kmeans_cpu_update(const float *points_aos, const int *assignments, float *centroids, float *sums, int *counts, int N, int D, int K, float *out_shift_sq);`
     * `static inline void kmeans_cpu_reference(const float *points_aos, const float *init_centroids, float *final_centroids, int *assignments, int N, int D, int K, int max_iters, bool fixed_iters, float tol, KMeansMetrics *metrics);`
     * `static inline VerifyResult verify_kmeans_outputs(const float *ref_centroids, const int *ref_labels, const float *test_centroids, const int *test_labels, int N, int D, int K);`
     * `static inline int dump_reference_files(const char *prefix, const float *centroids, const int *labels, int N, int D, int K);`
     * `static inline int load_reference_files(const char *prefix, float **centroids, int **labels, int N, int D, int K);`
     * `static inline void compute_derived_metrics(KMeansMetrics *m, int N, int D, int K);`
     * `static inline double median_double(double *arr, int n);` (using C `qsort`)
     * `static inline int append_csv_row(const char *csv_path, const char *impl_name, const KMeansConfig *cfg, const KMeansMetrics *m, const VerifyResult *v);`
   * `verify_kmeans_outputs` exact metrics:
     $$\text{centroid\_rel\_l2} = \frac{\sqrt{\sum_{c=0}^{K-1} \sum_{d=0}^{D-1} (\mu_{c,d}^{\text{test}} - \mu_{c,d}^{\text{ref}})^2}}{\max\!\left(10^{-12}, \sqrt{\sum_{c=0}^{K-1} \sum_{d=0}^{D-1} (\mu_{c,d}^{\text{ref}})^2}\right)}, \qquad \text{centroid\_max\_abs} = \max_{c,d} \left|\mu_{c,d}^{\text{test}} - \mu_{c,d}^{\text{ref}}\right|$$
     $$\text{label\_match\_pct} = 100.0 \times \frac{1}{N} \sum_{i=0}^{N-1} \mathbb{I}\!\left[a_i^{\text{test}} = a_i^{\text{ref}}\right]$$
     * `v.passed = (v.centroid_rel_l2 <= VERIFY_REL_L2_TOL) && (v.label_match_pct >= VERIFY_LABEL_MATCH_MIN);`
   * `src/kmeans_seq.c`: Parses CLI, loads/generates data, validates dataset, initializes centroids, runs `warmup` untimed + `trials` timed invocations of `kmeans_cpu_reference`, takes median timings, prints summary, and handles `--verify`, `--dump-ref`, `--check-ref`, and `--csv`.
3. **Implementation Steps:**
   - [ ] Implement `include/verify.h` and `src/kmeans_seq.c`.
   - [ ] Build `make bin/kmeans_seq bin/kmeans_seq_prof`.
   - [ ] Run deterministic dump and check self-test on $N = 10{,}000, D = 18, K = 16$.
4. **Verification & Acceptance Criteria:**
   - [ ] Both targets compile with `gcc -std=c11 -Wall -Wextra` and zero warnings.
   - [ ] `./bin/kmeans_seq --synthetic --n 10000 --d 18 --k 16 --iters 20 --seed 42 --dump-ref data/ref_10k` exits `0`.
   - [ ] Re-running with `--check-ref data/ref_10k` asserts `centroid_rel_l2 == 0.000000e+00`, `centroid_max_abs == 0.000000e+00`, and `label_match_pct == 100.0000%`.

---

#### Task 0.4: Profile Serial Baseline with `gprof` / `perf` to Prove Hot Spot
**Status:** `NOT_STARTED` | **Completed (UTC):** `—`

1. **Objective & Requirement Mapping:** Satisfies **REQ-04** ("Baseline first, then profile; never optimize blind").
2. **Exact Technical Decisions:**
   * Workload: $N = 1{,}000{,}000, D = 18, K = 32, I = 20$ (on `data/susy.bin` if present, else `--synthetic`).
   * Run `bin/kmeans_seq` to capture exact phase timings (`assign_ms` vs. `update_ms`) and compute assignment percentage.
   * Run `bin/kmeans_seq_prof` followed by `gprof bin/kmeans_seq_prof gmon.out` (if `gprof` is installed) and remove `gmon.out`.
   * If `perf` is permitted (`perf stat`), also capture `cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses`.
3. **Implementation Steps:**
   - [ ] Execute serial profiling on `bin/kmeans_seq` and `bin/kmeans_seq_prof`.
   - [ ] Write combined output (`wall_seconds` phase breakdown, `gprof` flat profile, and optional `perf stat`) to `results/baseline_profile.txt`.
   - [ ] Remove temporary `gmon.out`.
4. **Verification & Acceptance Criteria:**
   - [ ] `results/baseline_profile.txt` exists and proves distance assignment consumes $> 90\%$ of total compute time (expected: $97\text{--}99\%$).

---

### Phase 1: Progressive C-Style CUDA Implementations

#### Task 1.1: Build Pure-C CUDA Driver (`main_cuda.c`) & Stage 1 Naive CUDA Kernel (`kmeans_cuda_naive.cu`)
**Status:** `NOT_STARTED` | **Completed (UTC):** `—`

1. **Objective & Requirement Mapping:** Satisfies **REQ-01**, **REQ-05**, **REQ-06**, **REQ-09**. Establishes the pure-C11 CUDA harness and Stage 1 global-memory baseline.
2. **Exact Technical Decisions:**
   * `include/kmeans_cuda.h`:
     ```c
     #ifndef KMEANS_CUDA_H
     #define KMEANS_CUDA_H

     #include "kmeans_types.h"

     #ifdef __cplusplus
     extern "C" {
     #endif

     const char *kmeans_cuda_impl_name(void);

     void kmeans_cuda_run(
         const float *h_points_aos,
         const float *h_init_centroids,
         float *h_final_centroids,
         int *h_assignments,
         const KMeansConfig *cfg,
         KMeansMetrics *metrics);

     #ifdef __cplusplus
     }
     #endif

     #endif /* KMEANS_CUDA_H */
     ```
   * `src/main_cuda.c` (Pure C11, compiled with `gcc`): Parses CLI, loads/generates data, validates inputs, initializes centroids, executes `warmup` + `trials` calls to `kmeans_cuda_run`, computes median metrics via `median_double()`, runs `--verify` / `--check-ref` / `--dump-ref`, prints summary, and appends `--csv` row.
   * `src/kmeans_cuda_naive.cu` (Strict C-style CUDA; zero STL/classes/references):
     * Define `#define CUDA_CHECK(call)` exiting with `EXIT_CUDA_ERR` (`4`).
     * `const char *kmeans_cuda_impl_name(void) { return "cuda_naive"; }`
     * One-time setup (`setup_ms`): `cudaMalloc` for `d_points_aos`, `d_centroids`, `d_assignments`; copy `h_points_aos` and `h_init_centroids` H2D.
     * Per-iteration loop:
       1. Launch `assign_naive_kernel<<<grid, block>>>` (1 thread/point `i < N`, reading `d_points_aos[(size_t)i * D + d]` and `d_centroids[(size_t)c * D + d]` directly from global memory), timed via `cudaEvent` into `assign_ms`.
       2. `cudaMemcpy(h_assignments, d_assignments, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost)` (timed in `xfer_ms`).
       3. Recompute centroids on host CPU with deterministic empty-cluster rule (timed via `wall_seconds()` in `update_ms`).
       4. `cudaMemcpy(d_centroids, h_final_centroids, (size_t)K * D * sizeof(float), cudaMemcpyHostToDevice)` (timed in `xfer_ms`).
     * Set `total_compute_ms = assign_ms + update_ms + xfer_ms`.
3. **Implementation Steps:**
   - [ ] Write `include/kmeans_cuda.h`, `src/main_cuda.c`, and `src/kmeans_cuda_naive.cu`.
   - [ ] Build via `make bin/kmeans_cuda_naive`.
   - [ ] Run verification command on $N = 100{,}000, D = 18, K = 32$.
4. **Verification & Acceptance Criteria:**
   - [ ] `obj/main_cuda.o` compiles cleanly with `gcc -std=c11 -Wall -Wextra`, and `bin/kmeans_cuda_naive` links cleanly.
   - [ ] `./bin/kmeans_cuda_naive --synthetic --n 100000 --d 18 --k 32 --iters 20 --block-size 256 --verify` exits `0` with `verify_status=PASS`, `centroid_rel_l2 <= 1e-4`, `label_match_pct >= 99.99%`.

---

#### Task 1.2: Implement Stage 2 Coalesced Padded SoA & Tiled Shared-Memory Kernel (`kmeans_cuda_shmem.cu`)
**Status:** `NOT_STARTED` | **Completed (UTC):** `—`

1. **Objective & Requirement Mapping:** Satisfies **REQ-01**, **REQ-05**, **REQ-06**. Demonstrates global memory coalescing (128-float Padded SoA) and conflict-free shared-memory centroid broadcast with automatic 48 KB tiling.
2. **Exact Technical Decisions:**
   * File: `src/kmeans_cuda_shmem.cu` (`kmeans_cuda_impl_name()` returns `"cuda_shmem"`).
   * Host setup (`setup_ms`): Compute `N_pad = compute_padded_n(N)`, allocate pinned staging buffer `h_points_soa` via `cudaHostAlloc`, call `convert_aos_to_soa`, and copy `d_points_soa` H2D.
   * Assignment kernel `assign_shmem_tiled_kernel`:
     * `int k_tile = SHMEM_MAX_FLOATS / D; if (k_tile > K) k_tile = K;`
     * `size_t shmem_bytes = (size_t)k_tile * (size_t)D * sizeof(float);`
     * Loop `for (int c_base = 0; c_base < K; c_base += k_tile)` loading `c_count * D` floats cooperatively into `extern __shared__ float s_centroids[];`, syncing with `__syncthreads()`, reading coalesced `d_points_soa[(size_t)d * N_pad + i]` and broadcast `s_centroids[(size_t)c_local * D + d]`, and syncing with `__syncthreads()`.
   * Centroid update remains host-side (identical to Stage 1) so the delta between Stage 1 and Stage 2 isolates **pure assignment-kernel optimization**.
3. **Implementation Steps:**
   - [ ] Implement `src/kmeans_cuda_shmem.cu`.
   - [ ] Build via `make bin/kmeans_cuda_shmem`.
   - [ ] Verify both single-tile ($K = 32, D = 18$) and multi-tile ($K = 1024, D = 18$, exceeding 48 KB) runs with `--verify`.
4. **Verification & Acceptance Criteria:**
   - [ ] Both $K = 32$ and $K = 1024$ runs pass `--verify` with exit code `0`.
   - [ ] On $N = 500{,}000, D = 18, K = 32$, `assign_ms` of `kmeans_cuda_shmem` is strictly lower than `assign_ms` of `kmeans_cuda_naive`.

---

#### Task 1.3: Implement Stage 3 Optimized Full On-Device Kernel (`kmeans_cuda_device.cu`)
**Status:** `NOT_STARTED` | **Completed (UTC):** `—`

1. **Objective & Requirement Mapping:** Satisfies **REQ-01**, **REQ-05**, **REQ-06**. Combines `float4` 4-point thread-coarsened coalesced SoA loads, tiled shared-memory centroids, two-tier on-device centroid reduction, `__shfl_down_sync` warp-shuffle convergence reduction, and pinned asynchronous CUDA streams to eliminate per-iteration PCIe transfers.
2. **Exact Technical Decisions:**
   * File: `src/kmeans_cuda_device.cu` (`kmeans_cuda_impl_name()` returns `"cuda_device"`).
   * Host setup (`setup_ms`): Create `cudaStream_t stream`; allocate pinned host buffers via `cudaHostAlloc(..., cudaHostAllocDefault)` for SoA points and final outputs; perform one-time async H2D copy of `d_points_soa` and `d_centroids` on `stream`.
   * Persistent device buffers: `d_assignments` ($N$ ints), `d_new_centroids` ($K \cdot D$ floats), `d_counts` ($K$ ints), `d_shift_sq` (1 float).
   * Inside the $I$-iteration loop on `stream` (**zero per-iteration H2D/D2H transfers when `fixed_iters` is true**, so `xfer_ms = 0.0`):
     1. **Assignment (`assign_ms`):** Launch `assign_opt_float4_kernel<<<vec_grid, block, shmem_bytes, stream>>>` implementing Section B.5 (`float4` vectorized SoA loads + 4-point register reuse + tiled `__shared__` centroids).
     2. **Zero + Accumulate + Normalize (`update_ms`):**
        * `cudaMemsetAsync(d_new_centroids, 0, (size_t)K * D * sizeof(float), stream);`
        * `cudaMemsetAsync(d_counts, 0, (size_t)K * sizeof(int), stream);`
        * If `!cfg->fixed_iters`, `cudaMemsetAsync(d_shift_sq, 0, sizeof(float), stream);`
        * If `((size_t)K * D + (size_t)K) <= ACCUM_SHMEM_MAX_FLOATS` (`6144` floats = 24 KB): launch `accumulate_shmem_kernel<<<pt_grid, block, ((size_t)K * D + (size_t)K) * sizeof(float), stream>>>`.
        * Else: launch `accumulate_global_kernel<<<pt_grid, block, 0, stream>>>`.
        * Launch `normalize_centroids_kernel<<<(K * D + 255) / 256, 256, 0, stream>>>` with deterministic empty-cluster retention and Pascal-safe `warp_reduce_sum_f32()` (`__shfl_down_sync`) for `d_shift_sq`.
        * If `!cfg->fixed_iters`, async copy 4-byte `d_shift_sq` to pinned host scalar (timed in `xfer_ms`), synchronize `stream`, and break if `h_shift_sq <= tol * tol`.
   * After loop (timed in `setup_ms`): One-time async D2H copy of `d_centroids` and `d_assignments`, `cudaStreamSynchronize(stream)`, and cleanup.
3. **Implementation Steps:**
   - [ ] Implement `src/kmeans_cuda_device.cu`.
   - [ ] Build all binaries via `make clean && make all`.
   - [ ] Verify Path A ($K = 32, D = 18$), Path B ($K = 512, D = 18$), and multi-tile + Path B ($K = 1024, D = 18$) with `--verify`.
4. **Verification & Acceptance Criteria:**
   - [ ] All three verification runs exit `0` with `centroid_rel_l2 <= 1e-4` and `label_match_pct >= 99.99%`.
   - [ ] `xfer_ms` equals `0.000` in `--fixed-iters` mode.
   - [ ] On $N = 1{,}000{,}000, D = 18, K = 32$, `ms_per_iter` satisfies strict monotonic improvement:
     $$T_{\text{seq}} > T_{\text{cuda\_naive}} > T_{\text{cuda\_shmem}} > T_{\text{cuda\_device}}$$

---

### Phase 2: Correctness, Edge-Case & Determinism Suite

#### Task 2.1: Build Automated Verification & Edge-Case Test Runner (`tests/run_tests.sh`)
**Status:** `NOT_STARTED` | **Completed (UTC):** `—`

1. **Objective & Requirement Mapping:** Satisfies **REQ-06**. Exhaustively verifies mathematical accuracy, `float4` tail masking, deterministic tie-breaking, empty-cluster safety, boundary conditions, and invalid-input rejection across all 4 binaries.
2. **Exact Technical Decisions:**
   * Create `tests/run_tests.sh` executing 8 automated test groups and saving structured logs to `results/test_suite_summary.txt` and `results/tiny_check.txt`:
     1. **T1 — Tiny Exact Ground Truth:** Synthetic $N = 1024, D = 8, K = 4, I = 20$. All 4 binaries pass `--verify`.
     2. **T2 — Medium Realistic Parity:** Synthetic $N = 65536, D = 18, K = 32, I = 20$. All 4 binaries pass `--verify`.
     3. **T3 — Shared-Memory Tiling Boundary ($K \times D > 12288$):** Synthetic $N = 16384, D = 18, K = 1024, I = 5$. Exercises 2-pass centroid tiling and Global Atomic Path B across all CUDA binaries.
     4. **T4 — Prime Non-Multiple-of-128 & Non-Multiple-of-4 Size:** Synthetic $N = 10007$ (prime, `N % 4 == 3`, `N % 128 == 23`), $D = 13$ (prime), $K = 7, I = 10$. Explicitly validates `float4` tail masking and SoA padding across all 4 binaries.
     5. **T5 — Forced Empty-Cluster Edge Case:** Construct `data/test_empty_cluster.bin` ($N = 100, D = 4$: 50 copies of `(0,0,0,0)` and 50 copies of `(10,10,10,10)`) run with $K = 4$. Two clusters become empty on iteration 1. Assert zero `NaN`s and exact parity across all 4 binaries.
     6. **T6 — Exact Tie-Breaking Determinism:** Construct `data/test_ties.bin` where points lie strictly equidistant from centroids 0 and 1. Assert all 4 binaries assign index `0` identically.
     7. **T7 — Early-Stopping & Warp-Shuffle Convergence Test:** Run all 4 binaries with `--early-stop --tol 1e-4 --iters 100` on well-separated synthetic clusters ($N = 8192, D = 8, K = 8$). Assert `iters_run < 100`, matching `iters_run`, and matching centroids (directly testing `__shfl_down_sync` `d_shift_sq` reduction).
     8. **T8 — Invalid Input Rejection Tests (Negative Assertions):**
        * `N < K` (`--n 10 --k 20`) $\implies$ exit code `3`.
        * `D = 0` (`--d 0`) $\implies$ exit code `1` or `3`.
        * `block_size = 100` (not multiple of 32) $\implies$ exit code `1`.
        * `NaN` injected into `data/test_nan.bin` $\implies$ exit code `3`.
        * Missing file (`--file data/does_not_exist.bin`) $\implies$ exit code `2`.
3. **Implementation Steps:**
   - [ ] Implement fixture generation (`data/test_empty_cluster.bin`, `data/test_ties.bin`, `data/test_nan.bin`) inside `tests/run_tests.sh`.
   - [ ] Implement all 8 test groups with explicit PASS/FAIL counters in `tests/run_tests.sh` and `chmod +x tests/run_tests.sh`.
   - [ ] Execute `make test`.
4. **Verification & Acceptance Criteria:**
   - [ ] `make test` exits `0` with `ALL TESTS PASSED (8/8 groups)`.
   - [ ] `results/tiny_check.txt` and `results/test_suite_summary.txt` contain `PASS` for every implementation.

---

### Phase 3: Profiling, Benchmarking, Scaling & Bottleneck Analysis

#### Task 3.1: Execute Unified CPU & GPU Profiler Audit (`scripts/run_profiler.sh`)
**Status:** `NOT_STARTED` | **Completed (UTC):** `—`

1. **Objective & Requirement Mapping:** Satisfies **REQ-04** ("what the profiler said you fixed").
2. **Exact Technical Decisions:**
   * Profile workload: $N = 1{,}000{,}000, D = 18, K = 32, I = 10, \text{block\_size} = 256$ (on `data/susy.bin` if available, else `--synthetic`).
   * **CPU Section:** Runs `bin/kmeans_seq_prof` + `gprof` (and `perf stat` if permitted), logging to `results/baseline_profile.txt`.
   * **GPU Probe Hierarchy:**
     1. Check `command -v nvprof`: if available, run `nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency,dram_read_throughput,dram_write_throughput,shared_load_transactions_per_request,atomic_transactions` on all three CUDA binaries, plus `nvprof --print-gpu-trace` summary for H2D/D2H memcpy overhead.
     2. Else check `command -v nsys`: run `nsys profile --stats=true` on all three CUDA binaries to extract CUDA API, kernel time, and H2D/D2H memcpy breakdown.
     3. Check GPU compute capability via `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`; only invoke `ncu` if compute capability $\ge 7.0$ (never crash on Pascal `sm_61`).
     4. Always append the built-in `assign_ms`, `update_ms`, `xfer_ms`, `gflops`, `gbytes_s` comparison table into `results/cuda_profile.txt`.
3. **Implementation Steps:**
   - [ ] Write `scripts/run_profiler.sh` and `chmod +x scripts/run_profiler.sh`.
   - [ ] Execute `bash scripts/run_profiler.sh`.
4. **Verification & Acceptance Criteria:**
   - [ ] `results/cuda_profile.txt` exists and quantitatively shows Stage 1 $\to$ Stage 2 improvement in global load efficiency / `assign_ms`.
   - [ ] `results/cuda_profile.txt` quantitatively shows Stage 2 $\to$ Stage 3 elimination of per-iteration `xfer_ms` (`0.0 ms`) and host `update_ms`, plus lower `assign_ms` from `float4` register reuse.

---

#### Task 3.2: Execute Full Benchmark & Scaling Matrix (`scripts/run_benchmarks.sh`)
**Status:** `NOT_STARTED` | **Completed (UTC):** `—`

1. **Objective & Requirement Mapping:** Satisfies **REQ-07**, **REQ-08**. Populates all experimental datasets specified in Section E.
2. **Exact Technical Decisions:**
   * Implement `scripts/run_benchmarks.sh` to execute Suites B, C, D, and E with `--warmup 1 --trials 5 --fixed-iters --iters 20`.
   * Write 4 dedicated CSV files:
     * `results/suite_b_n_scaling.csv`
     * `results/suite_c_gpu_saturation.csv`
     * `results/suite_d_k_sweep.csv`
     * `results/suite_e_block_and_dim.csv`
   * Run one verification pass (`--verify`) on the largest SUSY configuration ($N = 5{,}000{,}000, D = 18, K = 32$) for `kmeans_cuda_device` and log to `results/susy_5m_check.txt`.
   * Runtime safeguard: if a single `kmeans_seq` run at $N = 5{,}000{,}000$ takes $> 15\text{ s}$, use `--warmup 0 --trials 3` for `kmeans_seq` at $N \ge 2{,}500{,}000$ only (keeping `--warmup 1 --trials 5` for all CUDA runs) and log this explicitly in Section F.
3. **Implementation Steps:**
   - [ ] Write `scripts/run_benchmarks.sh` and `chmod +x scripts/run_benchmarks.sh`.
   - [ ] Execute `bash scripts/run_benchmarks.sh` and check off each completed suite in Section E.
   - [ ] Validate all 4 CSV files for non-empty rows, zero `NaN`s, and exact 21-column formatting.
4. **Verification & Acceptance Criteria:**
   - [ ] All 4 CSV files exist in `results/` with row counts matching Section E.
   - [ ] `results/susy_5m_check.txt` records `PASS` (`centroid_rel_l2 <= 1e-4`, `label_match_pct >= 99.99%`).

---

#### Task 3.3: Generate Publication-Quality Dual PNG + Vector PDF Figures (`figures/*.py`)
**Status:** `NOT_STARTED` | **Completed (UTC):** `—`

1. **Objective & Requirement Mapping:** Satisfies **REQ-08**, **REQ-09**, **REQ-10**. Produces both 300 DPI PNGs (for Markdown/slides) and crisp vector PDFs (for ACM 2-column LaTeX).
2. **Exact Technical Decisions:**
   * All Python scripts use `matplotlib.use("Agg")`, font size $\ge 10\text{ pt}$, colorblind-safe Okabe-Ito palette, crisp gridlines, explicit axis units, and save **both** `.png` (300 DPI) and `.pdf` (vector) into `figures/`:
     1. `figures/fig_scaling.py`:
        * `figures/speedup_vs_n.{png,pdf}`: Log-x plot of speedup ($S = T_{\text{seq}} / T_{\text{GPU}}$) across $N \in \{100\text{K}, 500\text{K}, 1\text{M}, 2.5\text{M}, 5\text{M}\}$ comparing `cuda_naive`, `cuda_shmem`, and `cuda_device` (showing both compute-only and end-to-end curves, plus a horizontal dashed line showing the theoretical $6\times$ multicore i5-8400 upper bound for context).
        * `figures/gpu_saturation.{png,pdf}`: Throughput (points/sec and GFLOP/s) and per-point latency of `cuda_device` as $N$ scales from 1 SM wave ($7{,}168$ points) to full saturation ($10{,}000{,}000$ points).
     2. `figures/fig_k_and_block.py`:
        * `figures/scaling_vs_k.{png,pdf}`: Two-panel plot across $K \in \{8, 16, 32, 64, 128, 256, 512, 1024\}$ showing `ms_per_iter` and `gflops`, annotated with vertical dashed markers at the GTX 1080 Ti ridge point ($K \approx 33$), Stage 3 shared-memory accumulation switch ($K = 323$), and 48 KB centroid tiling cliff ($K = 682$).
        * `figures/block_size_tuning.{png,pdf}`: Grouped bar/line plot of `ms_per_iter` and `gflops` vs. `block_size` $\in \{64, 128, 256, 512, 1024\}$ across all three CUDA stages.
        * `figures/dim_scaling.{png,pdf}`: Throughput (`gflops`) and speedup vs. serial CPU across dimensionality $D \in \{8, 16, 32, 64\}$.
     3. `figures/fig_breakdown.py`:
        * `figures/time_breakdown.{png,pdf}`: Stacked bar chart decomposing per-iteration `assign_ms`, `update_ms`, and `xfer_ms` across `cuda_naive`, `cuda_shmem`, and `cuda_device` at $N = 1\text{M}$ and $N = 5\text{M}$.
     4. `figures/fig_roofline.py`:
        * `figures/roofline.{png,pdf}`: Log-log Roofline plot with GTX 1080 Ti memory roof ($484\text{ GB/s}$) and FP32 compute roof ($11{,}340\text{ GFLOP/s}$), plus Intel i5-8400 single-core memory roof ($41.6\text{ GB/s}$) and compute roof ($128\text{ GFLOP/s}$). Overlay measured `kmeans_seq`, `cuda_naive`, `cuda_shmem`, and `cuda_device` points for $K \in \{8, \dots, 1024\}$ at $\text{AI} = \frac{3 K D}{4 (D + 1)}$.
3. **Implementation Steps:**
   - [ ] Write `figures/fig_scaling.py`, `figures/fig_k_and_block.py`, `figures/fig_breakdown.py`, and `figures/fig_roofline.py`.
   - [ ] Execute all four Python plotting scripts without errors.
4. **Verification & Acceptance Criteria:**
   - [ ] All 7 PNG files ($> 20\text{ KB}$ each) exist in `figures/`.
   - [ ] All 7 matching vector PDF files exist in `figures/`.

---

### Phase 4: Documentation, Report, Presentation & Final Rubric Audit

#### Task 4.1: Write Reproducible `README.md`
**Status:** `NOT_STARTED` | **Completed (UTC):** `—`

1. **Objective & Requirement Mapping:** Satisfies **REQ-09**, **REQ-10**.
2. **Exact Technical Decisions:**
   * Document hardware tested, `gcc`/`nvcc` versions, pure-C11 build commands (`make all`, `make CUDA_ARCH=...`), UCI SUSY preparation (`scripts/prepare_susy.sh`), optional UCI Power CSV conversion flags, CLI examples for all 4 binaries, verification (`make test`), profiling (`scripts/run_profiler.sh`), full benchmark reproduction (`scripts/run_benchmarks.sh`), and figure generation.
3. **Implementation Steps:**
   - [ ] Write `README.md` following the structure of `hw1/SpMV/README.md`.
4. **Verification & Acceptance Criteria:**
   - [ ] Every command documented in `README.md` matches the exact paths and flags implemented in Phases 0–3.

---

#### Task 4.2: Author ACM 2-Column Final Report (`report/report.md` & `report/report.tex`) and Presentation Outline
**Status:** `NOT_STARTED` | **Completed (UTC):** `—`

1. **Objective & Requirement Mapping:** Satisfies **REQ-10** ($\ge 4$ pages ACM 2-column format excl. references; 4-minute presentation + 30 s Q&A).
2. **Exact Technical Decisions:**
   * Structure `report/report.tex` using `\documentclass[sigconf]{acmart}` (embedding vector PDFs from `../figures/*.pdf`) and mirror content in `report/report.md` (embedding `../figures/*.png`).
   * Mandatory six-section narrative populated strictly with **measured numbers from `results/*.csv` and `results/*_profile.txt`** (zero fabricated or placeholder numbers):
     1. **Problem & Motivation:** Lloyd's K-Means $\mathcal{O}(I \cdot N \cdot K \cdot D)$ complexity, arithmetic intensity analysis, real-world UCI SUSY workload.
     2. **Serial Baseline & Profiling Analysis:** Pure-C11 i5-8400 `-O3 -march=native` serial baseline ($T_1$), `gprof`/`perf` evidence proving distance assignment consumes $> 90\%$ of runtime.
     3. **Parallel CUDA Design & Optimizations:** Progression across Stage 1 (AoS global memory + host update), Stage 2 (128-float Padded SoA coalescing + conflict-free tiled shared-memory centroids), and Stage 3 (`float4` 4-point thread coarsening + two-tier on-device shared-atomic reduction + Pascal-safe `__shfl_down_sync` warp reduction + pinned async streams).
     4. **Experimental Setup:** Hardware table (i5-8400 + GTX 1080 Ti), datasets (UCI SUSY + synthetic), timing methodology (1 warmup + 5 median trials isolating I/O, setup, compute, and transfers), and verification tolerances ($\le 10^{-4}$ relative $L_2$, $\ge 99.99\%$ labels).
     5. **Results, Roofline & Scaling Analysis:** Embed all figures; quantify Amdahl's Law serial fraction $1 - p$ on Stage 1/Stage 2 PCIe transfers; analyze ridge-point transition ($K \approx 33$) and 48 KB shared-memory cliff ($K = 682$); include an analytical comparison against a theoretical ideal $6\times$ multicore CPU bound on the i5-8400.
     6. **Conclusion & Future Work:** Peak speedups achieved, key profiler lessons learned, and extensions (multi-GPU MPI+CUDA, GEMM formulation via $\|x - \mu\|^2 = \|x\|^2 + \|\mu\|^2 - 2 x \cdot \mu$, and datacenter Volta/Ampere evaluation on NCSU ARC).
   * Write `presentation/slides_outline.md`: crisp 4-slide talk script (1 minute/slide) matching the exact 4-minute time limit + anticipated 30-second Q&A talking points.
3. **Implementation Steps:**
   - [ ] Populate `report/report.md` from recorded CSVs and profiler dumps.
   - [ ] Populate `report/report.tex` in ACM 2-column format (`sigconf`).
   - [ ] Write `presentation/slides_outline.md`.
   - [ ] If `pdflatex` is available, compile `report/report.tex` and verify PDF page count is $\ge 4$ pages (excluding references).
4. **Verification & Acceptance Criteria:**
   - [ ] Zero `[TODO]` or placeholder values exist anywhere in `report/` or `presentation/`.
   - [ ] All referenced figure paths resolve to valid PNG/PDF files in `figures/`.

---

#### Task 4.3: Execute Final Pre-Submission Rubric Audit
**Status:** `NOT_STARTED` | **Completed (UTC):** `—`

1. **Objective & Requirement Mapping:** Final gate covering **REQ-01** through **REQ-10**.
2. **Exact Technical Decisions:**
   * Run automated checklist verification and record output in `results/final_audit.txt`.
3. **Implementation Steps & Verification Gates:**
   - [ ] Confirm zero `.cpp`/`.hpp`/`.cuh` files exist and zero `#include <omp.h>` references exist (`find src include -name '*.cpp' -o -name '*.hpp' -o -name '*.cuh'` is empty).
   - [ ] `make clean && make all` exits `0` using `gcc -std=c11` and `nvcc`.
   - [ ] `make test` exits `0`.
   - [ ] All 4 CSVs in `results/` are non-empty and valid 21-column tables.
   - [ ] All 7 PNGs ($> 20\text{ KB}$) and 7 PDFs exist in `figures/`.
   - [ ] No huge temporary files (`data/*.csv`, `data/*.bin`, `data/*.gz`) are staged for Moodle submission.
   - [ ] Write final audit summary to `results/final_audit.txt` confirming `10/10` requirements satisfied.

---

## Section E: Benchmark & Experiment Matrix

All timing experiments isolate pure per-iteration compute (`assign_ms`, `update_ms`, `xfer_ms`, `total_compute_ms`) from one-time `io_ms` and `setup_ms`. Every configuration runs **1 untimed warmup** + **5 timed trials** (reporting median; exception: `kmeans_seq` at $N \ge 2{,}500{,}000$ may use 0 warmup + 3 trials if single-run runtime exceeds $15\text{ s}$). All scaling suites use `--fixed-iters --iters 20 --seed 42`.

### Benchmark Execution Tracker
- [ ] **Suite B** (`results/suite_b_n_scaling.csv`): 20 configurations completed
- [ ] **Suite C1** (`results/suite_c_gpu_saturation.csv`): 6 SM-wave saturation points completed
- [ ] **Suite C2** (`results/suite_c_gpu_saturation.csv`): 9 large-scale stress points completed
- [ ] **Suite D** (`results/suite_d_k_sweep.csv`): $K \in \{8 \dots 1024\}$ sweep completed
- [ ] **Suite E1** (`results/suite_e_block_and_dim.csv`): 15 block-size tuning points completed
- [ ] **Suite E2** (`results/suite_e_block_and_dim.csv`): 16 dimensionality sweep points completed
- [ ] **5M Verification Gate** (`results/susy_5m_check.txt`): `PASS` recorded

### E.1 Suite B: Problem Size $N$ Scaling & Serial-vs-GPU Speedup (`results/suite_b_n_scaling.csv`)

| Parameter | Exact Values |
| :--- | :--- |
| Implementations | `kmeans_seq`, `kmeans_cuda_naive`, `kmeans_cuda_shmem`, `kmeans_cuda_device` |
| Dataset | UCI SUSY prefix slices (`--file data/susy.bin --n <N>`; fallback `--synthetic`) |
| Sweep $N$ | `100000, 500000, 1000000, 2500000, 5000000` |
| Fixed Parameters | $D = 18$, $K = 32$, $I = 20$, `block_size = 256` |
| Metrics Recorded | `io_ms`, `setup_ms`, `assign_ms`, `update_ms`, `xfer_ms`, `total_compute_ms`, `ms_per_iter`, speedup vs. `kmeans_seq` (compute-only & end-to-end), `gflops`, `gbytes_s` |

### E.2 Suite C: GPU Grid Saturation & Large-Scale Throughput (`results/suite_c_gpu_saturation.csv`)

| Sub-Experiment | Implementations | Workload Scaling Rule | Fixed Parameters |
| :--- | :--- | :--- | :--- |
| **C1: SM Wave Saturation (SUSY)** | `kmeans_cuda_device` | $N \in \{7168, 28672, 114688, 458752, 1835008, 5000000\}$ ($1\times, 4\times, 16\times, 64\times, 256\times$ one wave across 28 SMs at 256 threads/block) | $D = 18, K = 32, I = 20$, `block_size = 256` |
| **C2: Large-Scale Stress Test (Synthetic)** | `kmeans_cuda_naive`, `kmeans_cuda_shmem`, `kmeans_cuda_device` | $N \in \{1000000, 5000000, 10000000\}$ (`--synthetic`) | $D = 18, K = 32, I = 20$, `block_size = 256` |

### E.3 Suite D: Cluster Count $K$, Ridge Point & 48 KB Shared-Memory Cliff (`results/suite_d_k_sweep.csv`)

| Parameter | Exact Values |
| :--- | :--- |
| Implementations | `kmeans_seq` (for $K \le 256$, or all $K$ if runtime $< 60\text{ s}$), `kmeans_cuda_naive`, `kmeans_cuda_shmem`, `kmeans_cuda_device` |
| Dataset | UCI SUSY slice ($N = 1{,}000{,}000, D = 18$; fallback `--synthetic`) |
| Sweep $K$ | `8, 16, 32, 64, 128, 256, 512, 1024` |
| Critical Thresholds Tested | $K \approx 33$ (GTX 1080 Ti Roofline ridge point); $K = 323$ (Stage 3 accumulation Path A $\to$ Path B at 24 KB); $K = 682$ (48 KB centroid single-tile $\to$ multi-tile cliff) |
| Fixed Parameters | $I = 20$, `block_size = 256` |

### E.4 Suite E: CUDA Block Size Tuning & Dimensionality Sweep (`results/suite_e_block_and_dim.csv`)

| Sub-Experiment | Implementations | Swept Variable | Fixed Parameters |
| :--- | :--- | :--- | :--- |
| **E1: CUDA Block Size Sweep** | `kmeans_cuda_naive`, `kmeans_cuda_shmem`, `kmeans_cuda_device` | `block_size` $\in \{64, 128, 256, 512, 1024\}$ | UCI SUSY $N = 5{,}000{,}000, D = 18, K = 32, I = 20$ |
| **E2: Dimensionality ($D$) Sweep** | `kmeans_seq`, `kmeans_cuda_naive`, `kmeans_cuda_shmem`, `kmeans_cuda_device` | Synthetic $D \in \{8, 16, 32, 64\}$ | `--synthetic`, $N = 1{,}000{,}000, K = 32, I = 20$, `block_size = 256` |

---

## Section F: Agent Execution Log & Blocker Register

### F.1 Append-Only Execution Log

The agent MUST append one row per completed task (or major milestone) with measured key numbers so later tasks (especially Task 4.2 report writing) have a single verified source of truth.

| Timestamp (UTC) | Task ID | Status | Key Measured Result / Artifact Summary |
| :--- | :--- | :--- | :--- |
| `—` | `INIT` | `READY` | Plan Version 3.1 initialized; ready for Phase 0 Task 0.1. |

### F.2 Blocker & Deviation Register

Record any environment constraint (e.g., `nvprof` unavailable, offline `SYNTHETIC_FALLBACK`, `pdflatex` missing, or `kmeans_seq` trial reduction) or failing verification gate here.

| Timestamp (UTC) | Task ID | Issue / Deviation | Mitigation / Resolution | Resolved? |
| :--- | :--- | :--- | :--- | :--- |
| `—` | `—` | None yet. | — | — |