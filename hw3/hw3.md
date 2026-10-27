# [HW3] CUDA — SpMV on the GPU (100 points)

**Release:** Wed, Oct 21, 2026
**Due:** Fri, Oct 30, 2026 11:59pm ET

**Single-person assignment.**

## Introduction

Now SpMV on the **GPU** with CUDA. The classic question is **how to map rows to threads**: one **thread per row** (scalar) is simple but suffers warp divergence and uncoalesced access when row lengths vary; one **warp per row** (vector) coalesces the row and tolerates imbalance better. The **ELL** layout (from HW2) maps very naturally to the GPU.

Baseline + datasets as before (SuiteSparse **D6-6, dictionary28, Ga3As3H12, bfly, pkustk14, roadNet-CA**, or **other SuiteSparse matrices** of your choice as long as you **explain your reason** in the report). Run on an **ARC GPU node**; build with `nvcc`.

## Tasks

You implement three CUDA kernels (Tasks 1–3), then write them up in a **report** whose required contents are graded separately (Tasks 4–6).

### Code

**1. CSR scalar kernel — one thread per row (15 pts).** Implement `spmv-cuda-scalar.cu`: host/device allocation, transfer, launch, copy back, and verify against the CPU baseline.

**2. CSR-vector kernel — one warp per row (20 pts).** Implement `spmv-cuda-vector.cu`: assign one warp per row and reduce the row's partial products with `__shfl`. Coalesced row access tolerates row-length imbalance better than the scalar kernel.

**3. ELL kernel (20 pts).** Implement `spmv-cuda-ell.cu` using the ELL layout from HW2, which maps naturally to the GPU (column-major, one thread per row over the padded columns).

### Report

Back every analysis with figures and/or tables — a claim without a supporting plot or table earns little credit.

**1. GPU behavior (15 pts).** Analyze warp divergence and memory coalescing as a function of row-length variance; study the effect of block size (e.g. 64/128/256/512). Use the matrices to make the point (`roadNet-CA` vs `pkustk14`).

**2. Performance & roofline (25 pts).** From your measurements:

- GPU speedup vs your CPU baseline (HW1/HW2): `S = T_cpu / T_gpu`; report with and without H2D/D2H transfer.
- Strong scaling over launch config (block/grid); scalar vs vector vs ELL.
- **GPU roofline**: AI ≈ 0.17 flop/byte vs the GPU's peak bandwidth (e.g. ~900 GB/s on V100) — confirm memory-bound; report achieved vs peak BW.

**3. Write-up (5 pts).** Kernel design, the scalar-vs-vector-vs-ELL trade-off, and per-matrix results.

## Submission

Folder `HW3-unityID/` with your `.cu` files (`spmv-cuda-scalar.cu`, `spmv-cuda-vector.cu`, `spmv-cuda-ell.cu`), `Makefile`, and `report.pdf` (plots included); zip to `HW3-unityID.zip`.

## Grading

- CSR scalar kernel: **15**
- CSR-vector kernel (warp per row): **20**
- ELL kernel: **20**
- GPU behavior — divergence / coalescing / block-size: **15**
- Performance & roofline (speedup vs CPU, GPU roofline): **25**
- Write-up: **5**