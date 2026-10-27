/* HW3 Task 3: ELL SpMV on the GPU, one thread per row.
 *
 * REQUIRED, alongside spmv-cuda-scalar.cu and spmv-cuda-vector.cu.
 * Read include/spmv_cuda.h first, and include/formats_ell.h
 * for the layout.
 *
 * The point of this kernel: ELL is column-major, so at slot k the threads of a
 * warp read vals[k*num_rows + i] for 32 consecutive i -- one contiguous
 * 128-byte run, perfectly coalesced, with no row_ptr indirection and no
 * divergence, because every row has the same trip count K. The cost is that
 * every row does K slots' work whether or not it has K nonzeros, so a matrix
 * with a long tail pays for padding it never uses.
 *
 * You may reuse your HW2 CSR->ELL conversion; it belongs in setup, on the host,
 * before the upload.
 *
 * Build and test:
 *   make spmv_cuda_ell
 *   ./spmv_cuda_ell matrices/pkustk14.mtx --block=256
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "spmv_cuda.h"

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
    int max_row_len;          //K

    int   * d_col_idx;        //num_rows*K, column-major
    float * d_vals;           //num_rows*K, column-major
    float * d_x;
    float * d_y;

    cuda_stats stats;
} spmv_ctx;


/* TODO (Task 3): one thread per row over the column-major ELL.
 *
 * Thread i sums vals[k*num_rows + i] * x[col_idx[k*num_rows + i]] for
 * k in [0, K) and writes y[i]. Guard i < num_rows.
 *
 * Padded slots hold col_idx 0 and val 0.0f, so they multiply to an exact zero
 * and need no test in the inner loop -- the fixed trip count is the whole
 * point of the format.
 *
 * Note num_rows is the stride between slots. Use size_t for the offset:
 * num_rows*K overflows a 32-bit int on the larger matrices. */
__global__ void spmv_ell_kernel(int num_rows, int max_row_len,
                                const int * __restrict__ col_idx,
                                const float * __restrict__ vals,
                                const float * __restrict__ x,
                                float * __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows)
        y[i] = 0.0f;      /* placeholder: writes zeros, so the check FAILs */
}


extern "C" int spmv_cuda_setup(const csr_matrix * A, const cuda_opts * opts,
                               void ** ctx)
{
    spmv_ctx * c = (spmv_ctx*)calloc(1, sizeof(spmv_ctx));
    if (!c) return 1;

    c->num_rows     = A->num_rows;
    c->num_cols     = A->num_cols;
    c->num_nonzeros = A->num_nonzeros;

    /* TODO: build the column-major ELL on the host (K = csr_max_row_len(A);
     * entry (i,k) at k*num_rows + i; short rows padded with zeros), upload
     * col_idx and vals, and allocate d_x and d_y. Free the host copy once it is
     * on the device.
     *
     * num_rows*K can be far larger than nnz -- on Ga3As3H12 it is 16x the CSR
     * size -- so check both the host allocation and the cudaMalloc, and return
     * nonzero rather than crashing. */
    c->max_row_len = 0;

    /* TODO: launch configuration -- one thread per row, as in the scalar
     * kernel. */
    c->stats.block_size      = opts->block_size;
    c->stats.grid_size       = 0;
    c->stats.threads_per_row = 1;

    /* the harness checks max_row_len against the matrix, and uses the slot
     * count for the ELL storage and GB/s figures */
    c->stats.max_row_len = c->max_row_len;
    c->stats.ell_slots   = (long long)c->num_rows * (long long)c->max_row_len;

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
