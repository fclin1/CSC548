# Project Plan: Accelerating K-Means Clustering on GPUs (Pure C & CUDA)

**Project Title:** Accelerating K-Means Clustering on GPUs  
**Language & Paradigms:** Pure C (C99/C11) & CUDA C  
**Course:** CSC 548 · Parallel Systems (Fall 2026)  
**Author:** Frank Lin  
**Hardware Platforms:** Local NVIDIA GeForce GTX 1080 Ti (Pascal, 11GB VRAM, CUDA 12.0) & NCSU ARC GPU Cluster  
**Target Submission Portal:** Moodle  

---

## 1. Timeline & Deadlines

- **Nov 8, 2026:** Project kickoff & baseline formulation
- **Nov 16, 2026:** Target date for functional CUDA kernels & initial speedup results (ready for presentation slides)
- **Nov 18 / 23 / 30, 2026:** **In-Class Oral Presentation** (4-minute talk + 30-second Q&A, 10% of grade)
- **Dec 5, 2026:** Final scaling sweeps, profiling data collection, and report draft complete
- **Dec 7, 2026 (11:59 PM ET):** **Final Report + Code Submission** ($\ge 4$ pages ACM 2-column format + source code, 20% of grade)

---

## 2. Project Directory Structure

```text
CSC548/final/
├── data/                       # Datasets (UCI Household Power Consumption + synthetic)
├── include/                    # Pure C Header files
│   ├── kmeans.h                # Data structures and function prototypes
│   ├── timer.h                 # High-resolution clock_gettime timers
│   ├── io.h                    # CSV and binary I/O utilities
│   └── cuda_utils.h            # CUDA error checking macros and device helpers
├── src/                        # Pure C and CUDA C source files
│   ├── main.c                  # CLI driver and benchmark harness (pure C)
│   ├── kmeans_serial.c         # Single-threaded C baseline
│   ├── io.c                    # Standard C file reader / CSV parser
│   ├── verify.c                # Result verification & numerical assertion (pure C)
│   ├── kmeans_cuda_naive.cu    # Initial CUDA baseline (1 thread/point)
│   ├── kmeans_cuda_shmem.cu    # Shared memory centroid caching
│   └── kmeans_cuda_opt.cu      # Fully optimized (warp shuffle / on-GPU reductions)
├── scripts/                    # Automation scripts
│   ├── download_data.sh        # Fetch UCI power consumption dataset
│   ├── generate_synthetic.py   # Synthetic cluster generator (N, D, K sweeps)
│   ├── benchmark_sweep.py      # Automated benchmark runner across N, K, block sizes
│   └── plot_results.py         # Generate scaling and speedup plots
├── results/                    # CSVs and profiling logs (ncu / nsys / gprof)
├── report/                     # ACM 2-column LaTeX report source & figures
├── Makefile                    # Multi-target build configuration (gcc + nvcc)
├── Outline.md                  # Course instructions outline
├── Proposal.md                 # Original approved proposal
├── Plan.md                     # This plan
└── README.md                   # Build and reproduction instructions
```

---

## 3. Phased Implementation Roadmap

### Phase 1: Environment Setup, Datasets & Pure C Baseline (Nov 9 – Nov 11)
- [ ] Initialize repository directory structure (`src/`, `include/`, `data/`, `scripts/`, `results/`, `report/`).
- [ ] Implement data ingestion and normalization in pure C (`io.c` / `io.h`):
  - Standard C `fopen`, `getline`/`fgets`, and `strtof` parser for the *UCI Individual Household Electric Power Consumption* dataset (~2.07M points, 7 continuous features).
  - Robust handling of missing entries (`?` values in UCI dataset).
- [ ] Create synthetic data generator script (`generate_synthetic.py`) to easily test sweeps up to $N = 10^7$ points and $K \in [4, 64]$.
- [ ] Implement pure C serial Lloyd's K-Means (`kmeans_serial.c`):
  - Flat 1D contiguous allocations using standard `malloc` / `free` (`data[N * D]`, `centroids[K * D]`, `labels[N]`).
- [ ] Integrate high-resolution POSIX timers (`clock_gettime(CLOCK_MONOTONIC, ...)` in `timer.h`) isolating:
  - Distance computation & cluster assignment step.
  - Centroid recalculation step.
  - Overall convergence loop.
- [ ] Profile serial code with `gprof` or `perf` to confirm distance computation is the dominant hotspot ($>90\%$ runtime).

### Phase 2: Naive CUDA Kernel & Pure C Correctness Harness (Nov 12 – Nov 14)
- [ ] Design memory representations:
  - Flat contiguous row-major or column-major arrays suitable for both C host pointers and CUDA device pointers.
- [ ] Implement naive CUDA distance kernel (`kmeans_cuda_naive.cu`):
  - 1 thread per data point.
  - Compute Euclidean distances to all $K$ centroids stored in global memory.
  - Write nearest centroid index to `labels[i]`.
- [ ] Implement host-side C update step or simple global atomic update on GPU.
- [ ] Build automated verification suite in pure C (`verify.c`):
  - Compare cluster assignments between CPU and GPU.
  - Assert that converged centroid positions match within numerical precision:
    $$\max |C_{\text{GPU}} - C_{\text{CPU}}| < 10^{-4}$$
  - Enforce the course requirement to avoid *"wrong answers, fast"*.
- [ ] Record initial baseline speedup $S_p = T_{\text{CPU}} / T_{\text{GPU\_naive}}$.

### Phase 3: Incremental GPU Optimizations & Profiling (Nov 15 – Nov 17)
- [ ] **Optimization 1 (Shared Memory Staging):**
  - Load $K \times D$ centroids into `__shared__` memory per thread block to replace redundant global memory reads with fast on-chip broadcast reads.
- [ ] **Optimization 2 (Memory Coalescing & Alignment):**
  - Ensure global reads to point coordinates are coalesced across warps.
  - Test memory layouts (Structure-of-Arrays `SoA` vs Array-of-Structures `AoS`) and vectorized loads (`float4`).
- [ ] **Optimization 3 (On-GPU Centroid Reductions):**
  - Eliminate device-to-host memory roundtrips per iteration.
  - Implement block-level shared memory accumulation and warp shuffle reductions (`__shfl_down_sync`) followed by global `atomicAdd`.
- [ ] **Optimization 4 (Asynchronous Transfers & Pinned Memory):**
  - Use page-locked host memory (`cudaHostAlloc`) and CUDA streams.
- [ ] Profile each optimization step with Nsight Systems / Nsight Compute (`ncu`, `nsys`) to log DRAM throughput, compute utilization, and warp occupancy.

### Phase 4: Parameter Sweeps & Presentation Deck (Nov 18 – Nov 20)
- [ ] **Benchmarking Sweeps:**
  - **Thread Block Size:** Evaluate block sizes `64, 128, 256, 512, 1024`.
  - **Strong Scaling:** Fixed dataset size ($N = 2 \times 10^6$), compare CPU vs. Naive CUDA vs. Optimized CUDA.
  - **Weak Scaling / Dataset Size Sweep:** Scale $N$ from $10^4$ to $10^7$ points.
  - **Complexity Sweeps:** Vary cluster count $K \in \{4, 8, 16, 32, 64\}$ and feature dimensions $D \in \{2, 4, 8, 16\}$.
- [ ] **Prepare Oral Presentation (4-Minute Talk + 30s Q&A):**
  - Slide 1: Problem motivation, K-Means complexity $\mathcal{O}(I \cdot N \cdot K \cdot D)$, data parallelism suitability.
  - Slide 2: Parallel design (Naive vs Shared Memory vs Warp Shuffle Reductions).
  - Slide 3: Performance results (Speedup graphs, strong scaling, throughput).
  - Slide 4: Profiling takeaways, Amdahl analysis, bottlenecks, conclusion.
  - Rehearse to strictly respect the 4-minute limit.

### Phase 5: Presentation Delivery & Feedback Incorporation (Nov 21 – Nov 30)
- [ ] Deliver presentation in assigned class slot (Nov 18, 23, or 30).
- [ ] Integrate instructor and peer questions/feedback into experiments (e.g. running on ARC cluster nodes to compare Pascal GTX 1080 Ti vs. Volta/Ampere datacenter GPUs if accessible).

### Phase 6: ACM Final Report & Code Release (Dec 1 – Dec 7)
- [ ] Set up ACM 2-column LaTeX template in Overleaf.
- [ ] Author all required report sections ($\ge 4$ pages excluding references):
  1. Introduction & Motivation
  2. Serial Baseline & Profiling Hotspot Proof
  3. Parallel Design & CUDA Optimizations
  4. Experimental Setup & Hardware Specifications
  5. Speedup, Scaling & Roofline Analysis
  6. Discussion, Amdahl's Law Bottlenecks & Conclusion
- [ ] Produce publication-quality vector charts (PDF/EPS) for scaling and speedup.
- [ ] Finalize code repository, write comprehensive `README.md` and clean multi-target `Makefile` (`gcc` for C sources, `nvcc` for CUDA kernels).
- [ ] Submit final PDF report and source code archive on Moodle by Dec 7, 11:59 PM ET.

---

## 4. Key Metrics & Evaluation Criteria

- **Speedup Calculation:**
  $$S_p = \frac{T_{\text{serial}}}{T_{\text{parallel}}}$$
- **Correctness Tolerance:** Exact match of cluster labels or centroid coordinate divergence $\le 10^{-4}$.
- **Scaling Metrics:**
  - Strong scaling efficiency across varied thread block sizes and hardware configurations.
  - Throughput (data points processed per second).
- **Amdahl's Law Analysis:**
  $$S_{\text{latency}}(s) = \frac{1}{(1 - p) + \frac{p}{s}}$$
  Identify the exact serial fraction $(1 - p)$ restricting scaling (e.g. initialization, convergence delta calculation, host-device synchronization).
