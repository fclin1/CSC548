/* HW3 Task 2: CSR SpMV on the GPU, one WARP per row.
 *
 * REQUIRED, alongside spmv-cuda-scalar.cu and spmv-cuda-ell.cu.
 * Read include/spmv_cuda.h first.
 *
 * The point of this kernel: the 32 lanes of a warp walk one row together, so
 * consecutive lanes read consecutive vals/col_idx and the loads coalesce, and
 * a long row is shared by 32 threads instead of stalling one. The cost is a
 * cross-lane reduction per row, and 32 threads idling on a row with fewer than
 * 32 nonzeros. Which effect wins depends on the matrix -- analyze this in the
 * GPU behavior report.
 *
 * Build and test:
 *   make spmv_cuda_vector
 *   ./spmv_cuda_vector matrices/pkustk14.mtx --block=256
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "spmv_cuda.h"

#define WARP_SIZE 32

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
    int   * d_row_ptr;
    int   * d_col_idx;
    float * d_vals;
    float * d_x;
    float * d_y;
    cuda_stats stats;
} spmv_ctx;


/* TODO (Task 2): one warp per row.
 *
 *   warp = (blockIdx.x*blockDim.x + threadIdx.x) / WARP_SIZE   -> the row
 *   lane = threadIdx.x % WARP_SIZE                             -> 0..31
 *
 * Each lane sums vals[k]*x[col_idx[k]] for k = row_ptr[row]+lane,
 * stepping by WARP_SIZE to row_ptr[row+1]. Then reduce the 32 partial sums
 * with __shfl_down_sync (offsets 16, 8, 4, 2, 1) and have lane 0 write y[row].
 *
 * Two things to get right: guard row < num_rows, and remember blockDim.x must
 * be a multiple of WARP_SIZE for the lane arithmetic to hold -- the harness
 * already rejects a --block that is not.
 *
 * The reduction sums each row in a different order than the sequential
 * baseline, so a small nonzero relative error is expected. That is rounding. */
__global__ void spmv_csr_vector_kernel(int num_rows,
                                       const int * __restrict__ row_ptr,
                                       const int * __restrict__ col_idx,
                                       const float * __restrict__ vals,
                                       const float * __restrict__ x,
                                       float * __restrict__ y)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int row  = tid / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (row < num_rows && lane == 0)
        y[row] = 0.0f;    /* placeholder: writes zeros, so the check FAILs */
}


extern "C" int spmv_cuda_setup(const csr_matrix * A, const cuda_opts * opts,
                               void ** ctx)
{
    spmv_ctx * c = (spmv_ctx*)calloc(1, sizeof(spmv_ctx));
    if (!c) return 1;

    c->num_rows     = A->num_rows;
    c->num_cols     = A->num_cols;
    c->num_nonzeros = A->num_nonzeros;

    /* TODO: the same five device buffers and upload as in the scalar kernel. */

    /* TODO: the launch configuration. Careful -- you now need one WARP per row,
     * not one thread, so the grid covers num_rows*WARP_SIZE threads. Getting
     * this wrong is the classic way to silently compute only the first
     * 1/32 of y. */
    c->stats.block_size      = opts->block_size;
    c->stats.grid_size       = 0;
    c->stats.threads_per_row = WARP_SIZE;

    *ctx = c;
    return 0;
}


extern "C" void spmv_cuda_h2d(void * ctx, const float * x)
{
    (void)ctx; (void)x;    /* TODO */
}


extern "C" void spmv_cuda_kernel(void * ctx)
{
    (void)ctx;             /* TODO: launch, then synchronize */
}


extern "C" void spmv_cuda_d2h(void * ctx, float * y)
{
    (void)ctx; (void)y;    /* TODO */
}


extern "C" void spmv_cuda_stats(void * ctx, cuda_stats * out)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    *out = c->stats;
}


extern "C" void spmv_cuda_teardown(void * ctx)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    if (!c) return;
    /* TODO: cudaFree everything setup allocated. */
    free(c);
}
