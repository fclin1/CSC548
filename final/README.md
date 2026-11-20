# Parallel K-Means Clustering on NVIDIA GPUs: Pure C11 & CUDA

**Course:** CSC 548: Parallel Systems (Fall 2026)  
**Author:** Frank Lin (`fclin`)  
**Target Hardware:** Intel Core i5-9600K CPU + NVIDIA GeForce GTX 1080 Ti GPU (Pascal GP102, `sm_61`)  
**Dataset:** UCI SUSY Benchmark ($N = 5{,}000{,}000$ points, $D = 18$ features, $\approx 360\text{ MB}$ FP32)

---

## Overview

This repository provides a high-performance, deterministic implementation of Lloyd's K-Means clustering algorithm. It features a strict **Pure ISO C11** serial baseline and three progressive **CUDA C** acceleration stages:

1. **`kmeans_seq` (IMPL-0):** Compiler-optimized serial C11 baseline (`gcc -O3 -march=native`), profiling-verified ground truth.
2. **`kmeans_cuda_naive` (IMPL-1):** 1 thread/point, Row-Major Array-of-Structures (AoS) global memory layout, host CPU update, demonstrating memory coalescing and PCIe transfer bottlenecks.
3. **`kmeans_cuda_shmem` (IMPL-2):** 128-float Padded Structure-of-Arrays (SoA) enabling 100% coalesced 128-byte DRAM transactions + tiled on-chip shared-memory centroid broadcasts (0 bank conflicts).
4. **`kmeans_cuda_device` (IMPL-3):** Vectorized `float4` loads (`LDG.E.128`) with 4-way register reuse + two-tier on-device hierarchical atomic reductions + Pascal-safe `__shfl_down_sync` warp-shuffle convergence + pinned asynchronous CUDA streams (zero per-iteration PCIe transfers).

---

## Repository Tree

```text
final/
├── Makefile                    # Pure C11 & nvcc build rules (no OpenMP, no C++ STL)
├── README.md                   # This reproducibility documentation
├── .gitignore                  # Git ignore rules for data and binaries
├── include/
│   ├── alloc.h                 # 64-byte aligned posix_memalign memory helpers
│   ├── cmdline.h               # Unified CLI argument parser
│   ├── config.h                # Hardware constants, tolerances, and exit codes
│   ├── input.h                 # SplitMix64 PRNG, Box-Muller generator, binary loader, AoS->SoA
│   ├── kmeans_cuda.h           # Pure C / extern "C" linkage interface
│   ├── kmeans_types.h          # Data models: KMeansConfig, KMeansMetrics, VerifyResult
│   ├── timer.h                 # CLOCK_MONOTONIC timing wrappers
│   └── verify.h                # Ground truth verification, L2/label parity, CSV writer
├── src/
│   ├── csv_to_bin.c            # Streaming CSV to binary converter (UCI SUSY & Power)
│   ├── kmeans_seq.c            # Serial baseline executable
│   ├── main_cuda.c             # Pure C11 CLI harness for all CUDA stages
│   ├── kmeans_cuda_naive.cu    # Stage 1 Naive CUDA implementation
│   ├── kmeans_cuda_shmem.cu    # Stage 2 Padded SoA + Shared Memory implementation
│   └── kmeans_cuda_device.cu   # Stage 3 Vectorized float4 + On-Device reduction
├── scripts/
│   ├── prepare_susy.sh         # Downloads and converts UCI SUSY to data/susy.bin
│   ├── run_profiler.sh         # Executes gprof and GPU phase breakdown audit
│   └── run_benchmarks.sh       # Runs full parameter sweep matrix (Suites B, C, D, E)
├── tests/
│   ├── generate_fixtures.c     # Deterministic test fixture generator
│   └── run_tests.sh            # 8-group automated test suite
├── figures/
│   ├── fig_scaling.py          # Plots speedup vs N and GPU saturation
│   ├── fig_k_and_block.py      # Plots K-scaling, block tuning, and D-scaling
│   ├── fig_breakdown.py        # Plots per-iteration stacked time breakdowns
│   └── fig_roofline.py         # Plots empirical Roofline vs hardware ceilings
├── results/                    # Benchmark CSVs, logs, and profiler reports
├── report/                     # ACM 2-column final report draft
└── presentation/               # 4-minute in-class presentation deck outline
```

---

## Quickstart & Build Instructions

### Prerequisites
* Linux or WSL2 x86_64 with NVIDIA CUDA Toolkit ($\ge 11.4$ or $12.x$)
* `gcc` ($\ge 9.0$, C11 support)
* `make`
* Python 3 with `numpy`, `pandas`, `matplotlib` (for offline figure generation)

### Compilation
```bash
# Build all executables (csv_to_bin, kmeans_seq, kmeans_seq_prof, kmeans_cuda_*)
make all

# Override CUDA architecture if running on Volta/Ampere/Hopper
make all CUDA_ARCH=sm_80
```

---

## Dataset Ingestion

To download and convert the real-world **UCI SUSY** dataset ($5{,}000{,}000$ rows, $18$ features):
```bash
bash scripts/prepare_susy.sh
```
This generates `data/susy.bin` ($360{,}000{,}016\text{ bytes}$) with a 16-byte binary header (`uint64_t N`, `uint64_t D`).

---

## Running the Executables

All executables share the exact same CLI flags:

```bash
# Run serial baseline on 100k synthetic points
./bin/kmeans_seq --synthetic --n 100000 --d 18 --k 32 --iters 20

# Run Stage 3 CUDA on real SUSY dataset with verification
./bin/kmeans_cuda_device --file data/susy.bin --n 1000000 --k 32 --iters 20 --block-size 256 --verify

# Dump reference outputs from serial baseline and check against GPU
./bin/kmeans_seq --file data/susy.bin --n 100000 --k 32 --iters 20 --dump-ref data/ref_100k
./bin/kmeans_cuda_device --file data/susy.bin --n 100000 --k 32 --iters 20 --check-ref data/ref_100k
```

---

## Verification & Testing Suite

Run the full 8-group automated regression and edge-case suite:
```bash
make test
# Or directly:
bash tests/run_tests.sh
```
This verifies:
1. `T1`: Tiny exact ground truth ($N=1024, D=8, K=4$).
2. `T2`: Medium realistic parity ($N=65536, D=18, K=32$).
3. `T3`: 48 KB shared-memory tiling cliff ($K=1024$).
4. `T4`: Prime and non-multiple-of-128 sizes ($N=10007, D=13, K=7$).
5. `T5`: Forced empty-cluster retention (zero NaNs).
6. `T6`: Deterministic tie-breaking behavior.
7. `T7`: Early-stopping convergence and warp-shuffle reduction.
8. `T8`: Negative input validation (rejecting out-of-bounds parameters).

---

## Reproducing Benchmarks & Profiler Audits

```bash
# Execute CPU and GPU profiler audit
bash scripts/run_profiler.sh

# Execute complete experimental scaling matrix (Suites B, C, D, E)
bash scripts/run_benchmarks.sh

# Generate all publication-quality PNG and vector PDF plots
python3 figures/fig_scaling.py
python3 figures/fig_k_and_block.py
python3 figures/fig_breakdown.py
python3 figures/fig_roofline.py
```
Outputs are saved into `results/*.csv` and `figures/*.{png,pdf}`.
