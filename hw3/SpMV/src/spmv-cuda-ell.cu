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
    if (i < num_rows) {
        float sum = 0.0f;
        for (int k = 0; k < max_row_len; k++) {
            size_t idx = (size_t)k * (size_t)num_rows + (size_t)i;
            sum += vals[idx] * x[col_idx[idx]];
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

    /* TODO: build the column-major ELL on the host (K = csr_max_row_len(A);
     * entry (i,k) at k*num_rows + i; short rows padded with zeros), upload
     * col_idx and vals, and allocate d_x and d_y. Free the host copy once it is
     * on the device.
     *
     * num_rows*K can be far larger than nnz -- on Ga3As3H12 it is 16x the CSR
     * size -- so check both the host allocation and the cudaMalloc, and return
     * nonzero rather than crashing. */
    int K = csr_max_row_len(A);
    c->max_row_len = K;

    /* TODO: launch configuration -- one thread per row, as in the scalar
     * kernel. */
    size_t slots = (size_t)A->num_rows * (size_t)K;

    int   * h_col_idx = NULL;
    float * h_vals    = NULL;

    if (slots > 0) {
        h_col_idx = (int*)calloc(slots, sizeof(int));
        h_vals    = (float*)calloc(slots, sizeof(float));
        if (!h_col_idx || !h_vals) {
            free(h_col_idx);
            free(h_vals);
            free(c);
            return 1;
        }

        int m = A->num_rows;
        for (int i = 0; i < m; i++) {
            int r_start = A->row_ptr[i];
            int r_end   = A->row_ptr[i + 1];
            int len     = r_end - r_start;
            for (int k = 0; k < len; k++) {
                size_t idx = (size_t)k * (size_t)m + (size_t)i;
                h_col_idx[idx] = A->col_idx[r_start + k];
                h_vals[idx]    = A->vals[r_start + k];
            }
        }
    }

    /* Allocate device buffers */
    cudaError_t err;
    if (slots > 0) {
        err = cudaMalloc((void**)&c->d_col_idx, slots * sizeof(int));
        if (err != cudaSuccess) {
            free(h_col_idx); free(h_vals); free(c);
            return 1;
        }
        err = cudaMalloc((void**)&c->d_vals, slots * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(c->d_col_idx);
            free(h_col_idx); free(h_vals); free(c);
            return 1;
        }
    }

    err = cudaMalloc((void**)&c->d_x, (size_t)A->num_cols * sizeof(float));
    if (err != cudaSuccess) {
        if (slots > 0) { cudaFree(c->d_col_idx); cudaFree(c->d_vals); }
        free(h_col_idx); free(h_vals); free(c);
        return 1;
    }

    err = cudaMalloc((void**)&c->d_y, (size_t)A->num_rows * sizeof(float));
    if (err != cudaSuccess) {
        if (slots > 0) { cudaFree(c->d_col_idx); cudaFree(c->d_vals); }
        cudaFree(c->d_x);
        free(h_col_idx); free(h_vals); free(c);
        return 1;
    }

    if (slots > 0) {
        err = cudaMemcpy(c->d_col_idx, h_col_idx, slots * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(c->d_col_idx); cudaFree(c->d_vals); cudaFree(c->d_x); cudaFree(c->d_y);
            free(h_col_idx); free(h_vals); free(c);
            return 1;
        }
        err = cudaMemcpy(c->d_vals, h_vals, slots * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(c->d_col_idx); cudaFree(c->d_vals); cudaFree(c->d_x); cudaFree(c->d_y);
            free(h_col_idx); free(h_vals); free(c);
            return 1;
        }
        free(h_col_idx);
        free(h_vals);
    }

    int block_size = opts->block_size;
    int grid_size  = (A->num_rows + block_size - 1) / block_size;

    c->stats.block_size      = block_size;
    c->stats.grid_size       = grid_size;
    c->stats.threads_per_row = 1;
    /* the harness checks max_row_len against the matrix, and uses the slot
     * count for the ELL storage and GB/s figures */
    c->stats.max_row_len     = c->max_row_len;
    c->stats.ell_slots       = (long long)c->num_rows * (long long)c->max_row_len;

    *ctx = c;
    return 0;
}


extern "C" void spmv_cuda_h2d(void * ctx, const float * x)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    cudaMemcpy(c->d_x, x, (size_t)c->num_cols * sizeof(float), cudaMemcpyHostToDevice);
}


extern "C" void spmv_cuda_kernel(void * ctx)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    spmv_ell_kernel<<<c->stats.grid_size, c->stats.block_size>>>(
        c->num_rows, c->max_row_len, c->d_col_idx, c->d_vals, c->d_x, c->d_y);
    cudaDeviceSynchronize();
}


extern "C" void spmv_cuda_d2h(void * ctx, float * y)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    cudaMemcpy(y, c->d_y, (size_t)c->num_rows * sizeof(float), cudaMemcpyDeviceToHost);
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
    if (c->d_col_idx) cudaFree(c->d_col_idx);
    if (c->d_vals)    cudaFree(c->d_vals);
    if (c->d_x)       cudaFree(c->d_x);
    if (c->d_y)       cudaFree(c->d_y);
    free(c);
}
