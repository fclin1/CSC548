# HW3 starter code — SpMV on the GPU (CUDA)

You implement **three required files**: `spmv-cuda-scalar.cu` (Code 1),
`spmv-cuda-vector.cu` (Code 2), and `spmv-cuda-ell.cu` (Code 3). Each implements
the same six functions. The harness and headers are provided and should not
be modified — the graders build your three sources against their own copy.
Submit all three `.cu` files, your `Makefile`, and `report.pdf` as specified
in `hw3.md`.

## What is here

Sources live in `src/`, headers in `include/`.

| file | what it is |
| --- | --- |
| `src/spmv-cuda-scalar.cu` | **your file (Task 1, required).** One thread per row, CSR. |
| `src/spmv-cuda-vector.cu` | **your file (Task 2, required).** One warp per row, CSR, `__shfl` reduction. |
| `src/spmv-cuda-ell.cu` | **your file (Task 3, required).** One thread per row, ELL. |
| `include/spmv_cuda.h` | the interface all of them implement. Read this first. |
| `include/formats_ell.h` | the column-major ELL layout, carried over from HW2. |
| `src/main_cuda.c` | the harness: loads the matrix, calls your code, checks the answer, times it with and without transfers, reports. |
| `src/spmv_csr.c` | sequential CSR baseline — your `T_cpu` for Report 2, if you want a single-core one. |
| `src/spmv.c` | the original sequential COO version, for comparison. |
| `include/` | MatrixMarket reader, command line, timers. |

No matrices ship with the starter code — you download the assigned ones into
`matrices/`. Reuse the HW1/HW2 datasets; `hw3.md` lists the matrices and the
conditions for choosing alternatives.

**The harness contains no CUDA.** `main_cuda.c` is host C and calls no CUDA API
at all: every `cudaMalloc`, `cudaMemcpy` and kernel launch in this assignment is
yours, which is what Task 1 asks for. Your `.cu` is compiled by `nvcc` and the
harness by `gcc`, so the six functions are declared `extern "C"` — keep them
that way or the link fails on mangled names.

## Build and run

```sh
make spmv_cuda_scalar                                  # Task 1
./spmv_cuda_scalar matrices/dictionary28.mtx --block=256

make spmv_cuda_vector                                  # Task 2
make spmv_cuda_ell                                     # Task 3
make spmv_csr && ./spmv_csr matrices/dictionary28.mtx  # sequential baseline
make all
make clean
```

Source filenames use hyphens (`spmv-cuda-*.cu`); build targets and executable
names use underscores (`spmv_cuda_*`).

For submission, put the three `.cu` files at the root of `HW3-unityID/`,
alongside a copy of this `Makefile` and your `report.pdf`. To reproduce that
build, place the submission folder next to the provided `SpMV/` directory and
run from `SpMV/`:

```sh
make -f ../HW3-unityID/Makefile KERNEL_DIR=../HW3-unityID all
```

The Makefile uses the provided `src/` and `include/` for the harness and
`KERNEL_DIR` for your submitted kernels. Grading builds the three source files
independently with the grader's build rules and provided harness.

The Makefile targets `sm_70` (V100), which is what ARC's GPU nodes are. On
another card, override it:

```sh
make spmv_cuda_scalar GPU_ARCH=sm_80
```

Flags on any of the CUDA binaries:

| flag | effect |
| --- | --- |
| `--block=N` | threads per block, a multiple of 32 up to 1024 (default 256) |
| `--iters=N` | fix the timed iteration count |
| `--no-check` | skip the correctness check |
| `--csv` | one machine-readable result line, handy for scripting your sweeps |

Every run prints `PASS` or `FAIL` against the sequential CSR baseline and exits
nonzero on `FAIL`.

**Do not expect a relative L2 error of exactly 0.** `nvcc` contracts `a*b+c`
into a single FMA by default, and the warp-per-row reduction sums each row in a
different order than the sequential baseline. A few times `1e-8` is rounding;
the tolerance is `1e-5`. An error near the tolerance is a real bug.

## What your code must do

Read `include/spmv_cuda.h`. In short:

- **`spmv_cuda_setup()` does all the allocation and uploads A**, works out the
  launch configuration from `opts->block_size`, and records it in the stats. It
  is timed separately and runs once.
- **`spmv_cuda_kernel()` launches and synchronizes, and does nothing else.**
  It runs hundreds of times inside the timed loop. If you forget to synchronize
  you will be timing the launch, not the kernel, and your GPU will look
  impossibly fast — the first thing to suspect when a number looks too good.
- **`spmv_cuda_h2d()` and `spmv_cuda_d2h()` move x and y.** They are separate
  from the kernel so the harness can time one SpMV both ways: kernel alone, and
  kernel plus both transfers. That pair of numbers is what Report 2 compares, so
  do not fold a transfer into the kernel call.
- **Fill in `spmv_cuda_stats()`** with the block and grid size you actually
  launched and the threads per row (1 or 32). The harness prints them and puts
  them in the CSV; the ELL kernel also reports `max_row_len` and `ell_slots`,
  which the harness checks against the matrix and uses for the ELL GB/s figure.
- **Check the return of every CUDA call.** The `CUDA_TRY` macro in each stub
  does it for you in setup. A failed `cudaMalloc` that goes unnoticed shows up
  as a wrong answer much later, and `Ga3As3H12` in ELL really can fail to fit.

Your file must not contain `main()` and must not read the matrix file.

## Getting numbers for the report

```sh
# block-size sweep (Report 1): the two matrices to contrast
for b in 64 128 256 512; do
  ./spmv_cuda_scalar matrices/roadNet-CA.mtx --block=$b --csv
  ./spmv_cuda_scalar matrices/pkustk14.mtx   --block=$b --csv
done

# scalar vs vector vs ELL, same matrix and block size
./spmv_cuda_scalar matrices/pkustk14.mtx --block=256 --csv
./spmv_cuda_vector matrices/pkustk14.mtx --block=256 --csv
./spmv_cuda_ell matrices/pkustk14.mtx --block=256 --csv

# the CPU baseline for the speedup in Report 2
./spmv_csr matrices/pkustk14.mtx
```

The harness reports **kernel only** and **with H2D + D2H** separately, plus the
transfer share as a percentage. Use the kernel-only time for the roofline — it
is the only one that reflects device bandwidth rather than PCIe — and report
both for the speedup, saying which you used for which. An SpMV called once from
a host program pays the transfers; one inside an iterative solver on the GPU
does not, and the gap between those two numbers is the point.

Back every analysis with figures and/or tables. The report covers GPU behavior
(15 points), performance and roofline (25), and the write-up (5); the three
code tasks are worth 15, 20, and 20 points respectively.
