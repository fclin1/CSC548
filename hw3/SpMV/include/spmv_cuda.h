/* HW3 student interface -- PROVIDED, DO NOT MODIFY.
 *
 * You implement the six functions at the bottom of this file, in THREE files:
 *
 *   spmv-cuda-scalar.cu   Task 1, required.  One thread per row, CSR.
 *   spmv-cuda-vector.cu   Task 2, required.  One warp per row, CSR, with a
 *                                            __shfl reduction.
 *   spmv-cuda-ell.cu      Task 3, required.  One thread per row, ELL.
 *
 * Submit all three kernels. Each is graded independently.
 *
 * The files are compiled and linked separately, one binary each, so each must
 * be complete on its own. Do not #include one from another.
 *
 * Everything else -- reading the matrix, converting it to CSR, generating x,
 * checking your answer against the sequential baseline, timing, and reporting
 * -- is done for you by main_cuda.c, which is host C code and calls no CUDA at
 * all. Every cudaMalloc, cudaMemcpy and kernel launch in this assignment is
 * yours (Task 1 asks for exactly that).
 *
 * Your file must not contain main().
 */

#pragma once

#include "formats.h"
#include "formats_ell.h"

/* main_cuda.c is compiled as C and your file as CUDA C++, so the six functions
 * need C linkage or the link fails on mangled names. */
#ifdef __cplusplus
extern "C" {
#endif


/* ------------------------------------------------------------------ *
 * How the harness was invoked.
 * ------------------------------------------------------------------ */
typedef struct cuda_opts
{
    int block_size;   //--block=N, threads per block. Sweep for the GPU behavior report.
} cuda_opts;


/* ------------------------------------------------------------------ *
 * What your launch looked like, for the GPU behavior and performance reports.
 *
 * The harness prints these and puts them in the CSV, and uses the ELL geometry
 * to work out device traffic for the roofline -- an ELL kernel moves padding
 * too, and the GB/s number has to say so.
 * ------------------------------------------------------------------ */
typedef struct cuda_stats
{
    int block_size;        //threads per block you launched with
    int grid_size;         //blocks you launched
    int threads_per_row;   //1 for a thread-per-row kernel, 32 for warp-per-row

    /* ELL only; leave both 0 in a CSR kernel. */
    int       max_row_len; //your K
    long long ell_slots;   //num_rows * K, padding included
} cuda_stats;


/* ------------------------------------------------------------------ *
 * The six functions you implement.
 *
 * They are split this way so the harness can time the kernel with and without
 * the PCIe transfers, which the performance report compares. Keep each one
 * doing only its own job: an upload hidden inside the kernel call would show up as the
 * kernel being slow and would make your own speedup numbers wrong.
 * ------------------------------------------------------------------ */

/* Called once, before any timing, and timed separately.
 *
 * Allocate device memory and upload everything that does not change from one
 * SpMV to the next: the matrix, in whatever layout your kernel wants (the ELL
 * kernel converts here -- see formats_ell.h; the conversion itself is HW2 Task
 * 1 and you may reuse yours), plus the device x and y buffers the calls below
 * use. Work out your launch configuration here too and record it in the stats.
 *
 * A is host memory and stays valid for the run, but it is HOST memory: your
 * kernel cannot read it.
 *
 * Return 0 on success, nonzero to abort the run -- a failed cudaMalloc, for
 * instance. Check the return of every CUDA call; a silent failure here shows up
 * as a wrong answer much later. */
int spmv_cuda_setup(const csr_matrix * A, const cuda_opts * opts, void ** ctx);

/* Upload x (A->num_cols floats, host) into your device x buffer. */
void spmv_cuda_h2d(void * ctx, const float * x);

/* Launch your kernel once, y = A*x, both already on the device.
 *
 * MUST synchronize before returning (cudaDeviceSynchronize, or an event you
 * wait on), or the harness will time the launch rather than the kernel and
 * report a GPU that is impossibly fast. This is the call in the timed loop, so
 * do nothing else in it -- no allocation, no transfer. */
void spmv_cuda_kernel(void * ctx);

/* Download your device y buffer into y (A->num_rows floats, host). */
void spmv_cuda_d2h(void * ctx, float * y);

/* Fill in how you launched. The harness zeroes *out first. */
void spmv_cuda_stats(void * ctx, cuda_stats * out);

/* Free every device and host allocation spmv_cuda_setup made. */
void spmv_cuda_teardown(void * ctx);


#ifdef __cplusplus
}
#endif
