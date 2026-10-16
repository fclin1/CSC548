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

void gemm(int n, const float * A, const float * B, float * C,
          const gemm_opts * opts)
{
    /* TODO: your OpenMP kernel with a vectorised micro-kernel inside. */
    (void)n; (void)A; (void)B; (void)C; (void)opts;
}
