# [HW1] MPI Programming — Distributed SpMV (CSR) (110 points)

**Due:** Friday, September 11, 2026, 11:59 PM

## Introduction

Sparse Matrix–Vector multiplication (**SpMV**), `y = A·x`, is a core kernel in scientific computing (the inner loop of iterative solvers like Conjugate Gradient (CG) and Generalized Minimal Residual (GMRES)), graph analytics (PageRank, for example, is repeated SpMV on the graph's transition matrix), and sparse deep learning — a pruned weight matrix times a single activation vector is SpMV, generalizing to sparse matrix–matrix (SpMM) once inputs are batched. Because `A` is stored without its zeros, SpMV has **irregular memory access** and is typically **memory-bound**, which makes it an excellent lens on parallel performance. SpMV is our running problem for HW1–HW4; each assignment attacks it with a different model.

In this first assignment you parallelize SpMV across nodes with **MPI**, using the **Compressed Sparse Row (CSR)** format.

### Sparse Matrix format

A sparse matrix stores only its nonzeros, and there are several standard ways to do so. The simplest is the **coordinate (COO)** format: a flat list of `(row, column, value)` triples, i.e. three arrays `row_idx`, `col_idx`, `vals`, each of length `nnz`. COO is the easiest format to build and edit and makes no assumption about ordering, but it keeps an explicit row index for *every* nonzero and offers no direct way to jump to a given row — you would have to scan the whole list. It is usually the format a matrix is *read in* as (MatrixMarket files are essentially COO), then converted to something denser for computation.

**Compressed Sparse Row (CSR)** compresses away COO's redundant row index. For an `m×n` matrix with `nnz` nonzeros, CSR uses three arrays: `vals` — the nonzero values (length `nnz`), `col_idx` — their column indices (length `nnz`), and `row_ptr` — the per-row start offsets (length `m+1`). Row `i`'s nonzeros are `vals[row_ptr[i] .. row_ptr[i+1]-1]` at columns `col_idx[row_ptr[i] .. row_ptr[i+1]-1]`, so `row_ptr[i+1] - row_ptr[i]` is the length of row `i`. By replacing COO's `nnz` row indices with just `m+1` offsets, CSR uses less memory and gives O(1) direct access to any row — which is exactly the row-at-a-time access SpMV needs, and why it is the standard format for this kernel.

We provide a sequential **CSR-SpMV baseline** in the course repo: `SpMV/src/spmv_csr.c`.

### What we provide, and what you write

You write **two files, `spmv_mpi_allgather.c` and `spmv_mpi_sparse.c`**, one per communication scheme (Tasks 1 and 2). Both implement the same four functions and differ only in how they move `x`. The starter code provides everything else — the driver, the MatrixMarket reader, COO→CSR conversion, the row/column partitioning, the distribution of `A`, generation of `x`, the correctness check against the sequential baseline, timing, and CSV reporting.

The two files are compiled and linked **separately**, into two binaries, and graded separately. Each must therefore be complete on its own: do not `#include` one from the other, and do not expect them to share anything at run time. Yes, the local CSR kernel will be nearly identical in both — that is fine and intended.

The interface is documented in `SpMV/include/spmv_mpi.h`; read it before you start, and read `SpMV/README.md` for how to build and run. In outline:

```c
int  spmv_mpi_setup(const csr_local *A, const vec_partition *xpart, MPI_Comm comm, void **ctx);
void spmv_mpi(void *ctx, const csr_local *A, const float *x_local, float *y_local, MPI_Comm comm);
void spmv_mpi_stats(void *ctx, comm_stats *out);
void spmv_mpi_teardown(void *ctx);
```

Neither file may:

- define `main()`;
- call `MPI_Init` or `MPI_Finalize` — the harness already initializes and finalizes MPI;
- read the matrix file — the harness loads, partitions, and distributes `A` and `x` for you;
- communicate over `MPI_COMM_WORLD` — always use the `comm` argument passed to your functions, never the global communicator.

We build your two files against our own copy of the harness, so anything you change in the other files is not part of your submission.

Build and run (see **Datasets** below for how to get a matrix first):

```bash
make spmv_mpi_allgather
mpirun -n 4 ./spmv_mpi_allgather matrices/dictionary28.mtx

make spmv_mpi_sparse
mpirun -n 4 ./spmv_mpi_sparse matrices/dictionary28.mtx
```

Every run prints `PASS` or `FAIL` against the sequential CSR baseline.

### Datasets

SuiteSparse (sparse.tamu.edu): **D6-6, dictionary28, Ga3As3H12, bfly, pkustk14, roadNet-CA**. You may also test on **other SuiteSparse matrices** of your choice, but you need to **explain your reason** in the report (e.g. to probe a different size, density, or row-length-variance regime).

**How to download.** Every matrix belongs to a *group*, and the download URL is `https://sparse.tamu.edu/MM/<group>/<name>.tar.gz`. You can find a matrix by typing its name into the search box at sparse.tamu.edu and clicking the **Matrix Market** link on its page, or fetch all six from the command line — each tarball unpacks to `<name>/<name>.mtx`:

```bash
cd SpMV && mkdir -p matrices && cd matrices
for gm in JGD_Homology/D6-6 Pajek/dictionary28 AG-Monien/bfly \
          PARSEC/Ga3As3H12 Chen/pkustk14 SNAP/roadNet-CA ; do
  name=${gm##*/}
  curl -L "<https://sparse.tamu.edu/MM/$gm.tar.gz>" | tar xz   # -> <name>/<name>.mtx
  mv "$name/$name.mtx" . && rm -rf "$name"                   # flatten to matrices/<name>.mtx
done
```

The group for each of the six (i.e. the `<group>/<name>` you need in the URL):

| matrix | group | rows × cols | nnz |
| --- | --- | --- | --- |
| D6-6 | `JGD_Homology` | 120,576 × 23,740 (rectangular) | 146,880 |
| dictionary28 | `Pajek` | 52,652 × 52,652 | 178,076 |
| bfly | `AG-Monien` | 49,152 × 49,152 | 196,608 |
| Ga3As3H12 | `PARSEC` | 61,349 × 61,349 | 5,970,947 |
| pkustk14 | `Chen` | 151,926 × 151,926 | 14,836,504 |
| roadNet-CA | `SNAP` | 1,971,281 × 1,971,281 | 5,533,214 |

Then point either binary at a downloaded matrix, e.g. `mpirun -n 4 ./spmv_mpi_allgather matrices/pkustk14.mtx`. One thing to watch: all but D6-6 are stored as **symmetric**, so the `.mtx` file holds only one triangle and its header line shows about *half* the `nnz` above — the reader expands the matrix to the full count on load, and that expanded count is what drives your CSR footprint and timings. These six span very different sparsity structures, so note each matrix's rows, nnz, and especially its **row-length variance**.

## Tasks

You implement two things (Tasks 1–2), then write them up in a **report** whose required contents are graded separately (Task 3).

**1. Parallel SpMV with MPI — allgather baseline (25 pts).** Implement `spmv_mpi_allgather.c` against the interface above, using **1-D row-block partitioning**: rank `r` owns a contiguous block of rows of `A` and computes the matching block of `y`. The harness has already given you that block; what you write is the communication plus the local kernel.

- Get a correct version working first with the simplest communication: **`MPI_Allgatherv` the whole `x`** onto every rank, then run the local CSR kernel.
- Note that `A` is partitioned by rows and `x` by columns. These are different splits when the matrix is not square — one of the six matrices is rectangular.
- Build any communication plan in `spmv_mpi_setup()`. `spmv_mpi()` runs inside the timed loop; setup is timed and reported separately.
- Main configuration: `N 8 -n 8` (one rank per node).

**2. Optimize the `x` communication — exchange only what each rank needs (30 pts).** The allgather moves the *whole* vector to *every* rank, most of which a given rank never touches. In `spmv_mpi_sparse.c`, replace it with a plan that sends each rank **only the `x` entries its local rows actually reference**. In `spmv_mpi_setup()`, inspect your local `col_idx` to determine which remote columns you need (and, symmetrically, which of your columns other ranks need), and set up the point-to-point (or neighbourhood-collective) exchange **once**. `spmv_mpi()` then moves only those entries each iteration.

- The plan must be built **once in `spmv_mpi_setup()`**, never rebuilt inside the timed `spmv_mpi()` — the harness reports setup time separately, so a plan hidden in the timed loop shows up as inflated per-iteration cost.
- Start from your working Task 1 file rather than from the stub: the local kernel and the `spmv_mpi_stats()` bookkeeping carry over unchanged, and only the communication differs. Keep both files — Task 1's binary is what the communication analysis in Task 3d compares against, so you need to be able to re-run it.

**3. Report (45 pts).** Your report must contain the four items below. Every analysis (3b–3d) must be backed by **figures and/or tables** — you are strongly encouraged to include more rather than fewer; a claim without a supporting plot or table earns little credit.

**3a. Write-up (10 pts).** Describe your **design decisions**, **results**, **per-matrix behavior**, and **any anomalies** in enough detail that we can follow what you did and why.

**3b. Format storage (5 pts).** For each matrix, report its storage footprint in CSR vs COO and explain the difference in one or two sentences. Both the harness and `spmv_csr` print these numbers; your job is to explain what drives the ratio, and why it differs so much across the six matrices.

**3c. Performance & Scaling (15 pts).** Run on `p = 1, 2, 4, 8, 16` ranks. From **your own measurements**, report:

- **Speedup** `S(p) = T1 / Tp` and **efficiency** `E(p) = S(p) / p` — plot vs `p` (**strong scaling**). State whether your `T1` is the sequential baseline or your MPI code on one rank.
- A **roofline**: SpMV's arithmetic intensity is ≈ `2 flops / 12 bytes ≈ 0.17 flop/byte`; place your measured GFLOP/s on the node's roofline and argue whether you are memory-bound.

Discuss **per-matrix** differences (e.g. why does `roadNet-CA` behave differently from `pkustk14`?). The harness prints the per-rank nonzero imbalance for each run — use it.

**3d. Communication analysis (15 pts).** For **each of your two schemes**, count the **number of messages** and the **total bytes** moved by one SpMV. Report these through `spmv_mpi_stats()` — the harness sums them across ranks and prints the total — and explain them. A collective's message count depends on how you model it (a ring allgather costs `p−1` rounds while a recursive-doubling allgather costs `⌈log₂ p⌉`), so **state the model you are counting under** (use the communication-cost and collective material from Topics 2 and 3). Compare allgathering the whole `x` against sending only the entries each rank needs, both in your analysis and against the measured numbers, and note **which matrices benefit most** — a matrix with local structure saves far more than one whose neighbours are scattered across ranks by the row-block split.

---

## **Extra: ARC access proof (10 pts).**

In your report, include a screenshot of your HW0 MPI "hello world" program running on an ARC compute node — the output must show on the **compute node's hostname** (not the login node), which proves you have registered for and can run jobs on the ARC cluster. See HW0 for instructions.

## Submission

Folder `HW1-unityID/` only containing:

- `spmv_mpi_allgather.c` — your allgather baseline (Task 1).
- `spmv_mpi_sparse.c` — your optimized implementation (Task 2).
- `report.pdf` — your write-up, including all plots and anything we need to interpret your results.

Both source files are built and graded independently; a missing file scores 0 for that task and does not affect the other. Keep the filenames exactly as given — we build them by name.

Zip to `HW1-unityID.zip`. Do **NOT** include a Makefile or any other source file; we supply the build.

## Grading

- Parallel SpMV, allgather baseline — **25** (compile 5 / runnable 10 / correctness 10)
- Optimized `x` communication (needs-only exchange, plan built once in `spmv_mpi_setup()`) — **30** (compile 10 / runnable 10 / correctness 10)
- Report — write-up (design decisions, results, per-matrix behavior, anomalies): **10**
- Report — format storage (CSR vs COO): **5**
- Report — performance & scaling (speedup/efficiency/roofline plots): **15**
- Report — communication analysis: **15**
- ARC access proof — a screenshot of your HW0 MPI program running on an ARC compute node (proves ARC registration): **10**