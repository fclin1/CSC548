/* HW2 Task 4: SIMD (AVX2) SpMV over CSR.
 *
 * ONE OF THE FOUR FILES YOU SUBMIT. Read include/spmv_omp.h for the contract.
 *
 * Expect this to be harder than the ELL version, and expect it to win less --
 * that contrast is what Task 4 is asking you to explain. CSR rows have
 * irregular lengths and are stored back to back, so there is no fixed trip
 * count to vectorise across rows: you vectorise ALONG one row, gather
 * x[col_idx[k]] eight at a time, reduce the eight lanes to one scalar, and
 * handle a per-row tail of len % 8. Short rows pay that setup for very little
 * vector work.
 *
 * Built with -march=native (AVX2 + FMA). Build and test:
 *   make spmv_simd_csr
 *   ./spmv_simd_csr matrices/pkustk14.mtx --threads=8
 */

#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include "spmv_omp.h"
#include "alloc.h"

typedef struct spmv_ctx
{
    const csr_matrix * A;
    spmv_partition     partition;
    int                nthreads;
    int              * start;
    spmv_stats         stats;
} spmv_ctx;


int spmv_omp_setup(const csr_matrix * A, const spmv_opts * opts, void ** ctx)
{
    spmv_ctx * c = (spmv_ctx*)calloc(1, sizeof(spmv_ctx));
    if (!c) return 1;

    c->A         = A;
    c->partition = opts->partition;
    c->nthreads  = opts->nthreads;

    c->start = (int*)alloc_array(c->nthreads + 1, sizeof(int));
    if (!c->start) { free(c); return 1; }

    /* TODO (Task 3b): the nonzero-based row split, as in spmv_omp_csr.c. */
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

    /* TODO (Task 4): vectorise along each row, threaded as in spmv_omp_csr.c.
     *
     * For row i over k in [row_ptr[i], row_ptr[i+1]), eight nonzeros at a time:
     *   v   = _mm256_loadu_ps(&vals[k])
     *   col = _mm256_loadu_si256(&col_idx[k])
     *   xv  = _mm256_i32gather_ps(x, col, 4)
     *   acc = _mm256_fmadd_ps(v, xv, acc)
     * then reduce acc's eight lanes to one float and add the len % 8 tail
     * scalar.
     *
     * The lane reduction changes the order in which the row is summed, so a
     * small nonzero relative error against the baseline is expected here. */

    /* Task 4 - AVX2 SIMD SpMV along row over CSR */
    if (c->partition == SPMV_PART_ROW) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            long long my_rows = 0;
            long long my_work = 0;

            #pragma omp for schedule(runtime)
            for (int i = 0; i < A->num_rows; i++) {
                int r_start = A->row_ptr[i];
                int r_end   = A->row_ptr[i + 1];
                int len     = r_end - r_start;
                float sum   = 0.0f;
                int k       = r_start;

                if (len >= 8) {
                    __m256 acc = _mm256_setzero_ps();
                    int k_vec_end = r_start + (len & ~7);
                    for (; k < k_vec_end; k += 8) {
                        __m256 v    = _mm256_loadu_ps(&A->vals[k]);
                        __m256i col = _mm256_loadu_si256((const __m256i*)&A->col_idx[k]);
                        __m256 xv   = _mm256_i32gather_ps(x, col, 4);
                        acc         = _mm256_fmadd_ps(v, xv, acc);
                    }
                    __m128 lo   = _mm256_castps256_ps128(acc);
                    __m128 hi   = _mm256_extractf128_ps(acc, 1);
                    __m128 s128 = _mm_add_ps(lo, hi);
                    s128        = _mm_hadd_ps(s128, s128);
                    s128        = _mm_hadd_ps(s128, s128);
                    sum         = _mm_cvtss_f32(s128);
                }
                for (; k < r_end; k++) {
                    sum += A->vals[k] * x[A->col_idx[k]];
                }
                y[i] = sum;
                my_rows++;
                my_work += (long long)len;
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

            for (int i = r_start; i < r_end; i++) {
                int k_start = A->row_ptr[i];
                int k_end   = A->row_ptr[i + 1];
                int len     = k_end - k_start;
                float sum   = 0.0f;
                int k       = k_start;

                if (len >= 8) {
                    __m256 acc = _mm256_setzero_ps();
                    int k_vec_end = k_start + (len & ~7);
                    for (; k < k_vec_end; k += 8) {
                        __m256 v    = _mm256_loadu_ps(&A->vals[k]);
                        __m256i col = _mm256_loadu_si256((const __m256i*)&A->col_idx[k]);
                        __m256 xv   = _mm256_i32gather_ps(x, col, 4);
                        acc         = _mm256_fmadd_ps(v, xv, acc);
                    }
                    __m128 lo   = _mm256_castps256_ps128(acc);
                    __m128 hi   = _mm256_extractf128_ps(acc, 1);
                    __m128 s128 = _mm_add_ps(lo, hi);
                    s128        = _mm_hadd_ps(s128, s128);
                    s128        = _mm_hadd_ps(s128, s128);
                    sum         = _mm_cvtss_f32(s128);
                }
                for (; k < k_end; k++) {
                    sum += A->vals[k] * x[A->col_idx[k]];
                }
                y[i] = sum;
                my_rows++;
                my_work += (long long)len;
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
    c->stats.nthreads = c->nthreads;
    *out = c->stats;
}


void spmv_omp_teardown(void * ctx)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    if (!c) return;
    free(c->start);
    free(c);
}
