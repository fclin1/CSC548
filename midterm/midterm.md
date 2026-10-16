# Midterm Project — Performance Engineering of GEMM (100 points)

**Due:** Fri, Oct 16, 2026 11:59pm ET *(see Course Schedule)* · **Single-person project.**

## Introduction

Dense matrix–matrix multiplication, `C = A·B` (**GEMM**), is the **compute-bound** counterpoint to the memory-bound SpMV you study in the homeworks. With `O(N³)` flops over `O(N²)` data, GEMM has high *arithmetic intensity* — done well it approaches the machine's peak FLOP/s; done naively it is starved by the cache. That gap is the whole project.

This is a **performance-archaeology** project: take a naive GEMM and drive it toward peak, **explaining every step with profiling and the roofline**, then scale it across nodes with MPI. We grade your *measured analysis on the real machine* far more than "it runs" — that is the part an LLM cannot do for you.

We provide a skeleton (the naive baseline, a harness that generates the problem, checks your result and times it, a `Makefile`, and a small STREAM): Midterm repo. **You write four files** — `gemm_tiled.c`, `gemm_omp.c`, `gemm_simd.c` (Part 1, each implementing the same `gemm()` from `include/gemm.h`) and `gemm_mpi.c` (Part 2, `include/gemm_mpi.h`). Everything else is fixed: the graders build your four files against their own copy of the skeleton. Its `README.md` has the build, the flags and what each stage is allowed to use.

> 💡 **Read before you start.** Use **sizes that fit and verify quickly** (below) — do *not* try 65,536. The harness checks every run against a reference unless you pass `--no-check`; one naive call at `N = 4096` takes minutes, so the naive curve may stop at 2048. `gemm()` computes **`C += A·B`** — accumulate, do not overwrite: the harness calls it twice and expects `2·A·B`, because SUMMA accumulates into `C` across its `p` steps. `N` is always a multiple of 32 and of nothing else, so handle a partial last tile. For MPI, the process count `P = p×p` must be a **perfect square** and `N/p` a multiple of 32; launch with `mpirun --bind-to none` or every rank's threads share one core. In Part 2 you will **reuse your optimized Part-1 kernel** as the local multiply. Use single-precision `float` throughout — the peak FLOP/s, arithmetic intensity, and the `sgemm` reference all assume FP32.
> 

## Part 1 — Single-node performance engineering (55 pts)

Start from the naive triple loop and climb the roofline. Report `GFLOP/s` at every stage.

**1a. Baseline + profiling (10).** The naive `ijk` GEMM is provided (`gemm_naive.c`); verify it and measure it. **Profile** it (`gprof`, `perf stat`, or VTune): report the hotspot and the **cache-miss behavior** (`perf stat -e cache-misses` or `valgrind --tool=cachegrind`). Explain *why* naive GEMM gets a small fraction of peak.

**1b. Cache blocking / tiling (15).** Reorder loops (`ikj`) and **block/tile** for cache reuse (`gemm_tiled.c`, one thread). Show what each of the two changes buys in `GFLOP/s` and why, in terms of the working set. Sweep tile sizes (`--tile=`) at more than one `N` and explain the best — and explain it if there is no best.

**1c. OpenMP (15).** Parallelize across cores (`gemm_omp.c`). Report, from your data: **speedup** `S(t)=T1/Tt`, **efficiency** `E=S/t` (strong scaling, plot vs threads), and an estimate of the **serial fraction** `f` (Amdahl).

**1d. SIMD (10).** Vectorize the micro-kernel (`gemm_simd.c`, the only file built with `-march=native`: AVX intrinsics, or `-O3 -march=native` + `restrict` and a vectorization report). Show how close you get to single-node peak, and account for the gap.

**Roofline.** Produce **one roofline plot** for the node (memory-bandwidth ceiling + compute ceiling) with a **measured point for each stage** (naive → tiled → +OpenMP → +SIMD) climbing toward peak. Classify each stage as memory- or compute-bound. Determine the two ceilings for the node you run on and report them: compute peak from the CPU spec (cores × clock × FP32 SIMD lanes × 2 for FMA — use `lscpu` for core count, clock, and AVX width) and memory bandwidth from the **STREAM** benchmark; state the CPU model and both ceiling values in your report. *(Optional: add an OpenBLAS/MKL `sgemm` point as the practical "peak.")*

## Part 2 — 2-D Distributed MPI+OpenMP GEMM (45 pts)

Implement **Stationary-C SUMMA** on a `p×p` process grid for **square** matrices. Matrix `C` stays put (no communication once distributed); `A` and `B` blocks are broadcast.

```
Input: A(N×N), B(N×N) on a p×p grid (P = p²);  block size b = N/p
Output: C = A·B
1. Build row & column communicators (MPI_Cart_create or MPI_Comm_split)
2. Process (i,j) owns A_ij, B_ij, C_ij (each b×b); set C_ij = 0
3. for k = 0 .. p-1:
     a. row-broadcast A_ik (owner column k) across process-row i
     b. col-broadcast B_kj (owner row k) across process-column j
     c. C_ij += A_ik × B_kj      // local GEMM = your optimized Part-1 kernel
4. (small N only) gather C and verify
```

**2a. Implementation (25).** Correct SUMMA using your Part-1 kernel for the local multiply (`gemm_mpi.c`; `make gemm_mpi` links it with your `gemm_simd.c`, `LOCAL_GEMM=omp` with your `gemm_omp.c`).

**2b. Communication + scaling (15).** From your data: count **messages** and **bytes** moved (fill in `gemm_mpi_stats()`; state the cost model you charge a broadcast under, and check your count against a formula in `N` and `p`); **strong scaling** (fixed `N`, `P = 4, 9, 16` — `S`, `E` vs ranks) and **weak scaling** (grow `N` with `p` at a fixed block size `b = N/p`, so memory per rank is constant — note what that does to flops per rank before you draw the ideal line). Discuss where communication starts to dominate, using the communication time the harness reports.

## Testing

- **Single-node:** `N = 512, 1024, 2048, 4096`. The harness verifies every run against its reference; `--no-check` skips it once a kernel is trusted.
- **Distributed, strong scaling:** `N = 1536, 3072, 6144`; `P = 4, 9, 16` (each `N/p` for `p = 2, 3, 4` is a multiple of 32, so blocks tile evenly).
- **Distributed, weak scaling:** `b = 768` fixed — `N = 1536, 2304, 3072` on `P = 4, 9, 16`.
- Sizes for quick checks: any multiple of 32 works single-node (`--n=160` takes milliseconds); `--n=384` divides over `p = 2, 3, 4`.

## Submission

Folder `Midterm-unityID/` containing **your four files — `gemm_tiled.c`, `gemm_omp.c`, `gemm_simd.c`, `gemm_mpi.c` — your plots (incl. the roofline) and `report.pdf`**. Zip to `Midterm-unityID.zip`. No `Makefile` and no other source: the graders drop your four files into their own copy of the skeleton and build them with its flags, so a change to any other file is not part of your submission. Each of the four is compiled on its own; if a later stage needs code from an earlier one, copy it in.

## Grading

Each part separates **Code**, **Analysis**, and the written **Report**, so the analysis and write-up carry their own weight (as in HW1). Analysis and report together are 45 of the 100 points.

- **Part 1 — Single-node (55):**
    - Code (30): baseline 5 · tiling 10 · OpenMP 10 · SIMD 5
    - Report:
        - write-up quality(5): design decisions and what limited each stage
        - Analysis (20): the roofline plot with per-stage memory-/compute-bound classification, the tile-size and strong-scaling analysis, and the serial fraction *f*
- **Part 2 — MPI SUMMA (45):**
    - Code (25): correct Stationary-C SUMMA using your optimized Part-1 kernel
    - Report
        - Write-up quality (5): design decisions and what limited each stage
        - Analysis (15): message and byte counts, plus strong- and weak-scaling analysis

## References

SUMMA and the stationary-C variant you implement in Part 2:

- SUMMA (the algorithm) — R. A. van de Geijn and J. Watts, "SUMMA: Scalable Universal Matrix Multiplication Algorithm," *Concurrency: Practice and Experience* 9(4):255–274, 1997. PDF (LAPACK Working Note 96): lawn96.pdf
- Stationary-A / B / C SUMMA — M. D. Schatz, R. A. van de Geijn, and J. Poulson, "Parallel Matrix Multiplication: A Systematic Journey," *SIAM J. Sci. Comput.* 38(6):C748–C781, 2016. PDF
    - Derives stationary-A, -B, and -C; Part 2 uses **stationary-C**.
- Tutorial — FLAME "Programming for High Performance": matrix–matrix multiplication on distributed-memory architectures. Link.