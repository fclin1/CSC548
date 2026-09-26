/* HW2 Task 4: SIMD (AVX2) SpMV over ELL.
 *
 * ONE OF THE FOUR FILES YOU SUBMIT. Read include/spmv_omp.h for the contract
 * and include/formats_ell.h for the ELL layout. As in spmv_omp_ell.c, the CSR ->
 * ELL conversion is yours (Task 1) and belongs in setup.
 *
 * This is the layout SIMD likes: column-major means slot k of eight
 * consecutive rows is eight contiguous floats, so the values and the column
 * indices are one vector load each and only x[col] needs a gather.
 *
 * Built with -march=native (AVX2 + FMA). Build and test:
 *   make spmv_simd_ell
 *   ./spmv_simd_ell matrices/pkustk14.mtx --threads=8
 */

#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include "spmv_omp.h"
#include "alloc.h"

typedef struct spmv_ctx
{
    ell_matrix     ell;
    spmv_partition partition;
    int            nthreads;
    int          * start;
    spmv_stats     stats;
} spmv_ctx;


int spmv_omp_setup(const csr_matrix * A, const spmv_opts * opts, void ** ctx)
{
    spmv_ctx * c = (spmv_ctx*)calloc(1, sizeof(spmv_ctx));
    if (!c) return 1;

    c->partition = opts->partition;
    c->nthreads  = opts->nthreads;

    /* TODO (Task 1): the same column-major ELL build as in spmv_omp_ell.c. */

    // initialize the ELL matrix dimensions
    c->ell.num_rows     = A->num_rows;
    c->ell.num_cols     = A->num_cols;
    c->ell.num_nonzeros = A->num_nonzeros;

    int K = csr_max_row_len(A);
    c->ell.max_row_len  = K;

    // allocate memory for col_idx and vals arrays in the ELL matrix
    size_t slots = (size_t)A->num_rows * (size_t)K;
    if (slots > 0) {
        c->ell.col_idx = (int*)calloc(slots, sizeof(int));
        c->ell.vals    = (float*)calloc(slots, sizeof(float));

        // error check: if either allocation fails, free both and return 1
        if (!c->ell.col_idx || !c->ell.vals) {
            free(c->ell.col_idx);
            free(c->ell.vals);
            free(c);
            return 1;
        }

        // fill the ELL matrix with values from the CSR matrix
        int m = A->num_rows;
        for (int i = 0; i < m; i++) {
            int r_start = A->row_ptr[i];
            int r_end   = A->row_ptr[i + 1];
            int len     = r_end - r_start;
            for (int k = 0; k < len; k++) {
                size_t idx = (size_t)k * (size_t)m + (size_t)i;
                c->ell.col_idx[idx] = A->col_idx[r_start + k];
                c->ell.vals[idx]    = A->vals[r_start + k];
            }
        }
    }

    c->start = (int*)alloc_array(c->nthreads + 1, sizeof(int));
    if (!c->start) { free(c); return 1; }
    for (int t = 0; t <= c->nthreads; t++)
        c->start[t] = (t == 0) ? 0 : A->num_rows;

    /* Task 3b - Nonzero-based row partitioning via binary search */
    c->start[0] = 0;
    c->start[c->nthreads] = A->num_rows;
    for (int t = 1; t < c->nthreads; t++) {
        long long target = (long long)t * (long long)A->num_nonzeros / (long long)c->nthreads;
        int low = 0, high = A->num_rows;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (A->row_ptr[mid] < target)
                low = mid + 1;
            else
                high = mid;
        }
        int r = low;
        if (r > 0) {
            long long diff_curr = (long long)A->row_ptr[r] - target;
            if (diff_curr < 0) diff_curr = -diff_curr;
            long long diff_prev = target - (long long)A->row_ptr[r - 1];
            if (diff_prev < 0) diff_prev = -diff_prev;
            if (diff_curr > diff_prev)
                r = r - 1;
        }
        if (r < c->start[t - 1])
            r = c->start[t - 1];
        c->start[t] = r;
    }

    *ctx = c;
    return 0;
}


void spmv_omp(void * ctx, const csr_matrix * A, const float * x, float * y)
{
    spmv_ctx * c = (spmv_ctx*)ctx;

    memset(c->stats.rows, 0, sizeof c->stats.rows);
    memset(c->stats.work, 0, sizeof c->stats.work);

    /* TODO (Task 4): eight rows per vector, over the same OpenMP threading as
     * spmv_omp_ell.c.
     *
     * For a block of eight rows starting at i, with m = num_rows:
     *   acc = _mm256_setzero_ps()
     *   for k in [0, K):
     *     v   = _mm256_loadu_ps(&vals[k*m + i])          eight values
     *     col = _mm256_loadu_si256(&col_idx[k*m + i])    eight columns
     *     xv  = _mm256_i32gather_ps(x, col, 4)           eight x entries
     *     acc = _mm256_fmadd_ps(v, xv, acc)
     *   _mm256_storeu_ps(&y[i], acc)
     *
     * The rows do not divide by eight in general, so finish the last
     * num_rows % 8 rows scalar, or mask them.
     *
     * Note the accumulation order: the eight lanes are independent partial
     * sums, and each y[i] here is summed over k in the same order as the
     * scalar version, so this one stays close to the baseline. Where a SIMD
     * kernel reorders a single row's sum -- as the CSR version must -- expect
     * a small nonzero relative error. That is rounding, not a bug. */

    /*Task 4 - Across-row AVX2 SIMD SpMV over ELL */
    const int m = c->ell.num_rows;
    const int K = c->ell.max_row_len;
    const float * vals    = c->ell.vals;
    const int   * col_idx = c->ell.col_idx;

    if (c->partition == SPMV_PART_ROW) {
        int num_blocks = (m + 7) / 8;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            long long my_rows = 0;
            long long my_work = 0;

            #pragma omp for schedule(runtime)
            for (int b = 0; b < num_blocks; b++) {
                int i = b * 8;
                if (i + 8 <= m) {
                    __m256 acc = _mm256_setzero_ps();
                    for (int k = 0; k < K; k++) {
                        size_t idx = (size_t)k * (size_t)m + (size_t)i;
                        __m256 v    = _mm256_loadu_ps(&vals[idx]);
                        __m256i col = _mm256_loadu_si256((const __m256i*)&col_idx[idx]);
                        __m256 xv   = _mm256_i32gather_ps(x, col, 4);
                        acc         = _mm256_fmadd_ps(v, xv, acc);
                    }
                    _mm256_storeu_ps(&y[i], acc);
                    my_rows += 8;
                    my_work += 8 * (long long)K;
                } else {
                    for (int r = i; r < m; r++) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; k++) {
                            size_t idx = (size_t)k * (size_t)m + (size_t)r;
                            sum += vals[idx] * x[col_idx[idx]];
                        }
                        y[r] = sum;
                        my_rows++;
                        my_work += (long long)K;
                    }
                }
            }

            c->stats.rows[tid] = my_rows;
            c->stats.work[tid] = my_work;
        }
    } else {
        #pragma omp parallel
        {
            int tid     = omp_get_thread_num();
            int r_start = c->start[tid];
            int r_end   = c->start[tid + 1];
            long long my_rows = 0;
            long long my_work = 0;

            int i = r_start;
            for (; i + 8 <= r_end; i += 8) {
                __m256 acc = _mm256_setzero_ps();
                for (int k = 0; k < K; k++) {
                    size_t idx = (size_t)k * (size_t)m + (size_t)i;
                    __m256 v    = _mm256_loadu_ps(&vals[idx]);
                    __m256i col = _mm256_loadu_si256((const __m256i*)&col_idx[idx]);
                    __m256 xv   = _mm256_i32gather_ps(x, col, 4);
                    acc         = _mm256_fmadd_ps(v, xv, acc);
                }
                _mm256_storeu_ps(&y[i], acc);
                my_rows += 8;
                my_work += 8 * (long long)K;
            }
            for (; i < r_end; i++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    size_t idx = (size_t)k * (size_t)m + (size_t)i;
                    sum += vals[idx] * x[col_idx[idx]];
                }
                y[i] = sum;
                my_rows++;
                my_work += (long long)K;
            }

            c->stats.rows[tid] = my_rows;
            c->stats.work[tid] = my_work;
        }
    }

    (void)x;
}


void spmv_omp_stats(void * ctx, spmv_stats * out)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    c->stats.nthreads    = c->nthreads;
    c->stats.max_row_len = c->ell.max_row_len;
    c->stats.ell_slots   = (long long)c->ell.num_rows * (long long)c->ell.max_row_len;
    *out = c->stats;
}


void spmv_omp_teardown(void * ctx)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    if (!c) return;
    delete_ell_matrix(&c->ell);
    free(c->start);
    free(c);
}
