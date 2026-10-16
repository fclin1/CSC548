/* Midterm 1d: the threaded kernel with a vectorised micro-kernel.
 *
 * ONE OF THE FOUR FILES YOU SUBMIT. Read include/gemm.h for the contract.
 *
 * Start from your gemm_omp.c. This is the only file built with -march=native,
 * so it is the only one where AVX2/FMA exists -- whether you write the
 * intrinsics yourself or let the compiler vectorise (-O3 -march=native,
 * `restrict`, and -fopt-info-vec to see what it did). Either way the report
 * has to show what changed and how close to peak it got.
 *
 * What the SIMD stage is really about is not the 8-wide multiply, it is
 * keeping a small block of C in registers across the whole k loop so that the
 * inner loop does FMAs, not loads and stores of C. A micro-kernel of a few
 * rows by one or two vector widths of columns, fed by a broadcast of A and a
 * vector load of B, is the standard shape; packing the slice of B (or A) it
 * reads into a contiguous buffer first removes the stride-n accesses.
 *
 * n is a multiple of 32, so the column loop has no vector tail; the row loop
 * may have one if your micro-kernel is, say, 6 rows tall.
 *
 * Expect a small nonzero relative error here (~1e-7): _mm256_fmadd_ps rounds
 * a*b+c once where the reference rounds twice. It passes; an error near the
 * tolerance (1e-4) is a bug.
 *
 * Build and test:
 *   make gemm_simd
 *   ./gemm_simd --n=4096 --threads=16
 */

#include <omp.h>
#include <immintrin.h>
#include "gemm.h"

static inline void micro_kernel_6x16(
    int n, int k_start, int k_end,
    const float * restrict A, const float * restrict B, float * restrict C,
    int i, int j)
{
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();

    const float * a0 = A + (size_t)(i + 0) * n;
    const float * a1 = A + (size_t)(i + 1) * n;
    const float * a2 = A + (size_t)(i + 2) * n;
    const float * a3 = A + (size_t)(i + 3) * n;
    const float * a4 = A + (size_t)(i + 4) * n;
    const float * a5 = A + (size_t)(i + 5) * n;

    for (int k = k_start; k < k_end; k++) {
        const float * bk = B + (size_t)k * n + j;
        __m256 b0 = _mm256_loadu_ps(bk);
        __m256 b1 = _mm256_loadu_ps(bk + 8);

        __m256 a_val;

        a_val = _mm256_set1_ps(a0[k]);
        c00 = _mm256_fmadd_ps(a_val, b0, c00);
        c01 = _mm256_fmadd_ps(a_val, b1, c01);

        a_val = _mm256_set1_ps(a1[k]);
        c10 = _mm256_fmadd_ps(a_val, b0, c10);
        c11 = _mm256_fmadd_ps(a_val, b1, c11);

        a_val = _mm256_set1_ps(a2[k]);
        c20 = _mm256_fmadd_ps(a_val, b0, c20);
        c21 = _mm256_fmadd_ps(a_val, b1, c21);

        a_val = _mm256_set1_ps(a3[k]);
        c30 = _mm256_fmadd_ps(a_val, b0, c30);
        c31 = _mm256_fmadd_ps(a_val, b1, c31);

        a_val = _mm256_set1_ps(a4[k]);
        c40 = _mm256_fmadd_ps(a_val, b0, c40);
        c41 = _mm256_fmadd_ps(a_val, b1, c41);

        a_val = _mm256_set1_ps(a5[k]);
        c50 = _mm256_fmadd_ps(a_val, b0, c50);
        c51 = _mm256_fmadd_ps(a_val, b1, c51);
    }

    #define ACCUM_C(r, c_reg0, c_reg1) \
        do { \
            float * c_row = C + (size_t)(i + r) * n + j; \
            _mm256_storeu_ps(c_row, _mm256_add_ps(_mm256_loadu_ps(c_row), c_reg0)); \
            _mm256_storeu_ps(c_row + 8, _mm256_add_ps(_mm256_loadu_ps(c_row + 8), c_reg1)); \
        } while (0)

    ACCUM_C(0, c00, c01);
    ACCUM_C(1, c10, c11);
    ACCUM_C(2, c20, c21);
    ACCUM_C(3, c30, c31);
    ACCUM_C(4, c40, c41);
    ACCUM_C(5, c50, c51);
    #undef ACCUM_C
}

static inline void micro_kernel_generic(
    int n, int k_start, int k_end,
    const float * restrict A, const float * restrict B, float * restrict C,
    int i, int j, int num_rows)
{
    __m256 c0[6], c1[6];
    for (int r = 0; r < num_rows; r++) {
        c0[r] = _mm256_setzero_ps();
        c1[r] = _mm256_setzero_ps();
    }

    for (int k = k_start; k < k_end; k++) {
        const float * bk = B + (size_t)k * n + j;
        __m256 b0 = _mm256_loadu_ps(bk);
        __m256 b1 = _mm256_loadu_ps(bk + 8);
        for (int r = 0; r < num_rows; r++) {
            __m256 a_val = _mm256_set1_ps(A[(size_t)(i + r) * n + k]);
            c0[r] = _mm256_fmadd_ps(a_val, b0, c0[r]);
            c1[r] = _mm256_fmadd_ps(a_val, b1, c1[r]);
        }
    }

    for (int r = 0; r < num_rows; r++) {
        float * c_row = C + (size_t)(i + r) * n + j;
        _mm256_storeu_ps(c_row, _mm256_add_ps(_mm256_loadu_ps(c_row), c0[r]));
        _mm256_storeu_ps(c_row + 8, _mm256_add_ps(_mm256_loadu_ps(c_row + 8), c1[r]));
    }
}

void gemm(int n, const float * A, const float * B, float * C,
          const gemm_opts * opts)
{
    const int tile = opts->tile > 0 ? opts->tile : 64;

    for (int kk = 0; kk < n; kk += tile) {
        const int kmax = (kk + tile < n) ? kk + tile : n;

        #pragma omp parallel
        {
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < n - (n % 6); i += 6) {
                for (int j = 0; j < n; j += 16) {
                    micro_kernel_6x16(n, kk, kmax, A, B, C, i, j);
                }
            }

            #pragma omp single
            {
                int rem_start = n - (n % 6);
                int rem_count = n % 6;
                if (rem_count > 0) {
                    for (int j = 0; j < n; j += 16) {
                        micro_kernel_generic(n, kk, kmax, A, B, C, rem_start, j, rem_count);
                    }
                }
            }
        }
    }
}
