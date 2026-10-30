/* HW3 Task 1: CSR SpMV on the GPU, one thread per row.
 *
 * REQUIRED, alongside spmv-cuda-vector.cu (Task 2) and
 * spmv-cuda-ell.cu (Task 3). Each file is built and graded on its
 * own, so each must be complete by itself.
 *
 * Read include/spmv_cuda.h first: it documents cuda_opts, cuda_stats and the
 * six functions, and it is the contract main_cuda.c holds you to.
 *
 * Rules:
 *   - no main(), no reading the matrix file
 *   - every cudaMalloc / cudaMemcpy / launch in this assignment is yours
 *   - allocate and upload A in setup, never in spmv_cuda_kernel(): that runs
 *     hundreds of times inside the timed loop
 *   - spmv_cuda_kernel() must synchronize before it returns, or you are timing
 *     the launch and not the kernel
 *
 * Build and test:
 *   make spmv_cuda_scalar
 *   ./spmv_cuda_scalar matrices/dictionary28.mtx --block=256
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "spmv_cuda.h"

/* Report and bail on the first failed CUDA call. A silent failure in setup
 * turns into a wrong answer much later, where it is far harder to find. */
#define CUDA_TRY(call)                                                        \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__,            \
                    #call, cudaGetErrorString(_e));                           \
            return 1;                                                         \
        }                                                                     \
    } while (0)

typedef struct spmv_ctx
{
    int num_rows, num_cols, num_nonzeros;

    /* device copies of A, x and y */
    int   * d_row_ptr;
    int   * d_col_idx;
    float * d_vals;
    float * d_x;
    float * d_y;

    cuda_stats stats;
} spmv_ctx;


/* TODO (Task 1): one thread per row.
 *
 * Thread i handles row i: sum vals[k]*x[col_idx[k]] over
 * k in [row_ptr[i], row_ptr[i+1]) and write y[i]. Guard i < num_rows -- the
 * last block is not full.
 *
 * Think about what the threads of one warp are doing while you write it. They
 * run in lockstep over rows i..i+31, whose lengths differ, so the warp takes as
 * long as its longest row (divergence), and at step k they read vals at 32
 * scattered offsets rather than one contiguous run (uncoalesced). That is what
 * the GPU behavior report asks you to analyze, and what the warp-per-row
 * kernel addresses. */
__global__ void spmv_csr_scalar_kernel(int num_rows,
                                       const int * __restrict__ row_ptr,
                                       const int * __restrict__ col_idx,
                                       const float * __restrict__ vals,
                                       const float * __restrict__ x,
                                       float * __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows) {
        int start = row_ptr[i];
        int end   = row_ptr[i + 1];
        float sum = 0.0f;
        for (int k = start; k < end; k++) {
            sum += vals[k] * x[col_idx[k]];
        }
        y[i] = sum;
    }
}


extern "C" int spmv_cuda_setup(const csr_matrix * A, const cuda_opts * opts,
                               void ** ctx)
{
    spmv_ctx * c = (spmv_ctx*)calloc(1, sizeof(spmv_ctx));
    if (!c) return 1;

    c->num_rows     = A->num_rows;
    c->num_cols     = A->num_cols;
    c->num_nonzeros = A->num_nonzeros;

    /* TODO (Task 1): allocate the five device buffers and upload A and nothing
     * else -- x is uploaded by spmv_cuda_h2d, y is written by the kernel.
     *
     * Sizes: row_ptr num_rows+1 ints, col_idx and vals num_nonzeros each,
     * d_x num_cols floats, d_y num_rows floats. Wrap every call in CUDA_TRY.
     * Note num_cols and num_rows differ on D6-6, which is rectangular. */

    /* TODO: work out the launch configuration for --block and record it. One
    * thread per row here, so you need enough blocks to cover num_rows. */

    CUDA_TRY(cudaMemcpy(c->d_row_ptr, A->row_ptr, (size_t)(A->num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(c->d_col_idx, A->col_idx, (size_t)A->num_nonzeros * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(c->d_vals,    A->vals,    (size_t)A->num_nonzeros * sizeof(float), cudaMemcpyHostToDevice));

    /* Launch configuration: one thread per row */
    int block_size = opts->block_size;
    int grid_size  = (A->num_rows + block_size - 1) / block_size;

    c->stats.block_size      = block_size;
    c->stats.grid_size       = grid_size;
    c->stats.threads_per_row = 1;
    c->stats.max_row_len     = 0;
    c->stats.ell_slots       = 0;

    *ctx = c;
    return 0;
}


extern "C" void spmv_cuda_h2d(void * ctx, const float * x)
{
    /* TODO: copy num_cols floats from host x into ((spmv_ctx*)ctx)->d_x. */
    spmv_ctx * c = (spmv_ctx*)ctx;
    cudaMemcpy(c->d_x, x, (size_t)c->num_cols * sizeof(float), cudaMemcpyHostToDevice);
}


extern "C" void spmv_cuda_kernel(void * ctx)
{  
    /* TODO: launch spmv_csr_scalar_kernel with your configuration, then
     * synchronize. Nothing else belongs here -- no allocation, no transfer. */
    spmv_ctx * c = (spmv_ctx*)ctx;
    spmv_csr_scalar_kernel<<<c->stats.grid_size, c->stats.block_size>>>(
        c->num_rows, c->d_row_ptr, c->d_col_idx, c->d_vals, c->d_x, c->d_y);
    cudaDeviceSynchronize();
}


extern "C" void spmv_cuda_d2h(void * ctx, float * y)
{  
    /* TODO: copy num_rows floats from ((spmv_ctx*)ctx)->d_y back into host y. */
    spmv_ctx * c = (spmv_ctx*)ctx;
    cudaMemcpy(y, c->d_y, (size_t)c->num_rows * sizeof(float), cudaMemcpyDeviceToHost);
}


extern "C" void spmv_cuda_stats(void * ctx, cuda_stats * out)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    /* a CSR kernel leaves max_row_len and ell_slots at 0 */
    *out = c->stats;
}


extern "C" void spmv_cuda_teardown(void * ctx)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    if (!c) return;
    /* TODO: cudaFree everything setup allocated. */
    if (c->d_row_ptr) cudaFree(c->d_row_ptr);
    if (c->d_col_idx) cudaFree(c->d_col_idx);
    if (c->d_vals)    cudaFree(c->d_vals);
    if (c->d_x)       cudaFree(c->d_x);
    if (c->d_y)       cudaFree(c->d_y);
    free(c);
}
