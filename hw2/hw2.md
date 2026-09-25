# [HW2] OpenMP + SIMD — Node-Level SpMV (CSR/ELL, Load Balancing, Vectorization) (100 points)

**Release**: Wed, Sep 16, 2026
**Due:** Fri, Sep 25, 2026 11:59pm ET
**Single-person assignment.**

## Introduction

We stay with **SpMV** (`y = A·x`) but attack it at the **node level** with two axes of parallelism at once: **shared-memory threading (OpenMP)** and **data-level parallelism (SIMD)**. Along the way we add a second storage format, **ELL**, so we can study how data layout interacts with parallelism, load balance, and vectorization. One lesson runs through the whole assignment: SpMV is **memory-bound**, so more cores and wider vectors help far less than their raw counts suggest — and you will measure exactly that.

### ELL format

ELLPACK (ELL) format pads every row to the length of the **longest** row and stores the result as two dense `m × K` arrays (`col_idx`, `vals`), typically **column-major**. Access is regular and vectorization-friendly; but when the **max row length ≫ average**, the padding wastes memory and work. That tension drives both the load-balancing and the vectorization parts of this assignment.

### What we provide, and what you write

You write **four files**, one per kernel (Tasks 1-4). All four implement the same four functions and differ only in the storage format and in how the work is split. The starter code provides everything else — the driver, the MatrixMarket reader, COO→CSR conversion, generation of `x`, the correctness check against the sequential baseline, timing, the load-balance and padding numbers, and CSV reporting.

The four files are compiled and linked **separately**, into four binaries, and graded separately. Each must therefore be complete on its own: do not `#include` one from another, and do not expect them to share anything at run time. The local kernel will look similar in more than one of them — that is fine and intended.

The interface is documented in `SpMV/include/spmv_omp.h`; read it before you start, and read `SpMV/README.md` for how to build and run. In outline:

```c
int  spmv_omp_setup(const csr_matrix *A, const spmv_opts *opts, void **ctx);
void spmv_omp(void *ctx, const csr_matrix *A, const float *x, float *y);
void spmv_omp_stats(void *ctx, spmv_stats *out);
void spmv_omp_teardown(void *ctx);
```

None of the four files may:

- define `main()`;
- read the matrix file — the harness loads and converts `A` and generates `x` for you;
- call `omp_set_num_threads()` or `omp_set_schedule()` — the harness has already applied `-threads` and `-schedule` before it calls you;
- rebuild per iteration what belongs in `spmv_omp_setup()` — the ELL conversion and the per-thread row ranges are built once there, and setup is timed and reported separately.

The `make` rules build the SIMD files with `-march=native` (AVX2 + FMA) and the OpenMP files without it, so keep intrinsics out of `spmv_omp_*.c`. Build and run on an x86 compute node, not the login node.

**One difference from HW1:** your two SIMD kernels will not be bit-identical to the sequential baseline, and are not meant to be. `_mm256_fmadd_ps` rounds `a*b+c` once where the scalar code rounds twice, and the CSR version also reduces eight lanes per row; expect a relative L2 error between `1e-8` and `1e-6`, against a tolerance of `1e-5`. Your two *OpenMP* kernels should still be exact — no row is split between threads, and an ELL sweep adds each row's terms in the same order the CSR baseline does — so a nonzero error there means you reordered something.

## Datasets

SuiteSparse (sparse.tamu.edu): **D6-6, dictionary28, Ga3As3H12, bfly, pkustk14, roadNet-CA** — the same six as HW1. You may also test on **other SuiteSparse matrices** of your choice, but you need to **explain your reason** in the report (e.g. to probe a different size, density, or row-length-variance regime).

No matrices ship with the starter code. Download the six into `SpMV/matrices/` exactly as in HW1 — each tarball unpacks to `<name>/<name>.mtx`:

```bash
cd SpMV && mkdir -p matrices && cd matrices
for gm in JGD_Homology/D6-6 Pajek/dictionary28 AG-Monien/bfly \
          PARSEC/Ga3As3H12 Chen/pkustk14 SNAP/roadNet-CA ; do
  name=${gm##*/}
  curl -L "https://sparse.tamu.edu/MM/$gm.tar.gz" | tar xz   # -> <name>/<name>.mtx
  mv "$name/$name.mtx" . && rm -rf "$name"                   # flatten to matrices/<name>.mtx
done
```

| matrix | group | rows × cols | nnz |
| --- | --- | --- | --- |
| D6-6 | `JGD_Homology` | 120,576 × 23,740 (rectangular) | 146,880 |
| dictionary28 | `Pajek` | 52,652 × 52,652 | 178,076 |
| bfly | `AG-Monien` | 49,152 × 49,152 | 196,608 |
| Ga3As3H12 | `PARSEC` | 61,349 × 61,349 | 5,970,947 |
| pkustk14 | `Chen` | 151,926 × 151,926 | 14,836,504 |
| roadNet-CA | `SNAP` | 1,971,281 × 1,971,281 | 5,533,214 |

All but D6-6 are stored symmetric, so the `.mtx` header shows about half the nnz above; the reader expands to the full count on load. For this assignment the number to look at per matrix is the **max vs average row length** — it decides both ELL's padding (Task 5) and how unbalanced a static row split is (Task 3).

## Tasks

### Code

**1. CSR → ELL conversion (5 pts).** Convert CSR → ELL: two dense `m × K` arrays, `K` the longest row length, short rows padded with explicit zeros. Use the **column-major** layout the starter code documents in `include/formats_ell.h` — entry `(i,k)` at `k*num_rows + i` — because that is what makes the SIMD version in Task 4 possible: slot `k` of eight consecutive rows is eight contiguous floats, so only `x[col]` needs a gather. Build it in `spmv_omp_setup()`, in both `spmv_omp_ell.c` and `spmv_simd_ell.c`. The harness checks your `K` against the matrix, so a wrong conversion is caught even if the SpMV happens to pass.

**2. Parallel SpMV with OpenMP (25 pts).** Parallelize **both** `spmv_omp_csr.c` and `spmv_omp_ell.c` with OpenMP. Write the row-partitioned kernel with `schedule(runtime)` so `--schedule` selects the schedule without recompiling. Use `-N 1 -n 8` (8 threads) as the main configuration.

**3. Load balancing (20 pts).** The heart of the OpenMP part.

- Try `schedule(static)`, `schedule(dynamic, chunk)`, and `schedule(guided)` with a few chunk sizes.
- Compare **row-based** vs **nonzero-based** partitioning of work.
- On a high-variance matrix — `Ga3As3H12` **and** `pkustk14` **are the two to use** — **quantify** the imbalance (max vs average work per thread) and show how dynamic / nonzero partitioning fixes it. Report the per-thread rows and work through `spmv_omp_stats()`; the harness turns them into the max, the average and the imbalance ratio and prints them every run, so fill it in or you have nothing to plot.
- Fixing the imbalance and going faster are **not the same thing**, and you are expected to notice the difference. On one of those two matrices a dynamic schedule flattens the imbalance ratio but loses to plain `static` on wall clock, because the scheduling and the lost locality cost more than the idle threads did. Report both numbers, not just the one that agrees with you.

**4. SIMD vectorization (25 pts).** Implement a SIMD/AVX version of ELL-SpMV (`spmv_simd_ell.c`) and CSR-SpMV (`spmv_simd_csr.c`). For ELL-SpMV, process 8 rows per vector: with the column-major layout the values and column indices for rows `i..i+7` at slot `k` are one `_mm256_loadu_ps` / `_mm256_loadu_si256` each, then a **gather** (`_mm256_i32gather_ps`) for `x[col]` and `_mm256_fmadd_ps` into an accumulator, with a scalar/masked tail for the last `num_rows % 8` rows. For CSR-SpMV, expect this to be harder — rows have irregular lengths and are stored back-to-back, so you cannot pad to a fixed vector width; you vectorize *along* one row, gather `x[col_idx]` eight at a time, reduce the eight lanes to one scalar, and handle a per-row `len % 8` tail. Comment on where that hurts and how the result compares to ELL: a matrix whose average row is shorter than a vector pays the reduction on every row and can come out **slower** than the scalar version.

### Report

Back every analysis in this section with figures and/or tables — a claim without a supporting plot or table earns little credit.

**5. Format comparison — CSR vs ELL memory (5 pts).** For each matrix, report the memory consumption of CSR vs ELL and explain which matrices ELL wastes memory on and why (ELL pads every row to the longest, so compare max vs average row length).

**6. Performance, scaling & roofline (15 pts).** From your measurements:

- **Thread scaling:** speedup `S(t) = T1 / Tt` and efficiency `E(t) = S / t`, threads = 1, 2, 4, 8, 16 (strong-scaling plots).
- **SIMD speedup** `S = T_scalar / T_simd` for CSR and ELL.
- Compare **CSR vs ELL** across schedules and matrices — state which wins per matrix and **why**.
- **One combined roofline** (AI ≈ 0.17 flop/byte): place the scalar, OpenMP, and SIMD points; report **achieved bandwidth vs machine peak**; and explain *why* both threading and vectorization are limited (memory-bound). This reasoning, on **your** measured numbers, is the most important part of the grade.

**7. Write-up (5 pts).** Design decisions, the format trade-off, the load-balancing results, where SIMD did / did not help and why (the memory-bound discussion), per-matrix behavior, and any anomalies.

## Submission

Folder `HW2-unityID/` only containing:

- `spmv_omp_csr.c`, `spmv_omp_ell.c` — your OpenMP implementations.
- `spmv_simd_csr.c`, `spmv_simd_ell.c` — your SIMD/AVX implementations.
- `report.pdf` — your write-up, including all plots.

All four source files are built and graded independently; a missing file scores 0 for that kernel and does not affect the others. Keep the filenames exactly as given — we build them by name.

Zip to `HW2-unityID.zip`. Do **NOT** include a Makefile or any other source file; we supply the build.

## Grading

- CSR → ELL conversion: **5**
- OpenMP SpMV (CSR & ELL) — **25** (compile 5 / runnable 10 / correctness 10)
- Load balancing (schedules, partitioning, imbalance quantified): **20**
- SIMD vectorization (CSR & ELL) — **25** (compile 5 / runnable 10 / correctness 10)
- Report — format comparison (CSR vs ELL memory): **5**
- Report — performance, scaling & roofline (the memory-bound reasoning): **15**
- Report — write-up: **5**