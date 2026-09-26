/* HW2 Tasks 2-3: OpenMP SpMV over CSR.
 *
 * ONE OF THE FOUR FILES YOU SUBMIT. Its partners are spmv_omp_ell.c,
 * spmv_simd_csr.c and spmv_simd_ell.c; all four implement the same four
 * functions and are built and graded on their own, so each must be complete by
 * itself.
 *
 * Read include/spmv_omp.h first: it documents spmv_opts and spmv_stats, and it
 * is the contract main_omp.c holds you to.
 *
 * Rules:
 *   - no main(), no reading the matrix file
 *   - do not call omp_set_num_threads() or omp_set_schedule(): the harness has
 *     already applied --threads and --schedule
 *   - build anything reusable in spmv_omp_setup(); spmv_omp() runs hundreds of
 *     times inside the timed loop
 *
 * Build and test:
 *   make spmv_omp_csr
 *   ./spmv_omp_csr matrices/Ga3As3H12.mtx --threads=8 --partition=nnz
 */

#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "spmv_omp.h"
#include "alloc.h"

typedef struct spmv_ctx
{
    const csr_matrix * A;
    spmv_partition     partition;
    int                nthreads;

    /* Task 3, nonzero partitioning: thread t owns rows [start[t], start[t+1]).
     * Computed once here rather than per iteration. Length nthreads+1. */
    int * start;

    spmv_stats stats;
} spmv_ctx;


int spmv_omp_setup(const csr_matrix * A, const spmv_opts * opts, void ** ctx)
{
    spmv_ctx * c = (spmv_ctx*)calloc(1, sizeof(spmv_ctx));
    if (!c) return 1;

    c->A         = A;
    c->partition = opts->partition;
    c->nthreads  = opts->nthreads;

    /* alloc_array is malloc for arrays that may be empty -- see include/alloc.h */
    c->start = (int*)alloc_array(c->nthreads + 1, sizeof(int));
    if (!c->start) { free(c); return 1; }

    /* TODO (Task 3b): fill start[] so each thread gets about the same number
     * of NONZEROS rather than the same number of rows. Thread t should own the
     * rows around the nonzero index t*nnz/nthreads -- a binary search of
     * A->row_ptr finds the row boundary. Do not split a row between threads:
     * keeping whole rows means each y[i] is still accumulated by one thread in
     * one order, and no reduction is needed. */

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

    if (c->partition == SPMV_PART_ROW) {
        /* TODO (Task 2/3a): parallelise this over the rows. Use
         * schedule(runtime) so --schedule picks the schedule without a
         * recompile, and record per-thread rows/work into c->stats for the
         * Task 3 imbalance numbers.
         *
         * Row i's nonzeros are A->vals[k] at columns A->col_idx[k] for k in
         * [A->row_ptr[i], A->row_ptr[i+1]). Overwrite y[i]; do not accumulate. */

        /* Task 2 - OpenMP SpMV over CSR (Row-partitioned) */
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            long long my_rows = 0;
            long long my_work = 0;

            #pragma omp for schedule(runtime)
            for (int i = 0; i < A->num_rows; i++) {
                float sum = 0.0f;
                int r_start = A->row_ptr[i];
                int r_end   = A->row_ptr[i + 1];
                for (int k = r_start; k < r_end; k++) {
                    sum += A->vals[k] * x[A->col_idx[k]];
                }
                y[i] = sum;
                my_rows++;
                my_work += (long long)(r_end - r_start);
            }

            c->stats.rows[tid] = my_rows;
            c->stats.work[tid] = my_work;
        }
    } else {
        /* TODO (Task 3b): same kernel, but each thread walks its own row range
         * [c->start[t], c->start[t+1]) from setup -- one fixed range per
         * thread, so no schedule applies here. */
        for (int i = 0; i < A->num_rows; i++)
            y[i] = 0.0f;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int r_start = c->start[tid];
            int r_end   = c->start[tid + 1];
            long long my_rows = 0;
            long long my_work = 0;

            for (int i = r_start; i < r_end; i++) {
                float sum = 0.0f;
                int k_start = A->row_ptr[i];
                int k_end   = A->row_ptr[i + 1];
                for (int k = k_start; k < k_end; k++) {
                    sum += A->vals[k] * x[A->col_idx[k]];
                }
                y[i] = sum;
                my_rows++;
                my_work += (long long)(k_end - k_start);
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
    /* a CSR kernel leaves max_row_len and ell_slots at 0 */
    *out = c->stats;
}


void spmv_omp_teardown(void * ctx)
{
    spmv_ctx * c = (spmv_ctx*)ctx;
    if (!c) return;
    free(c->start);
    free(c);
}
