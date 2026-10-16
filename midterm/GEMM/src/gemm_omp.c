/* Midterm 1c: the tiled kernel, threaded with OpenMP.
 *
 * ONE OF THE FOUR FILES YOU SUBMIT. Read include/gemm.h for the contract.
 *
 * Start from your gemm_tiled.c (copy it in; the files are built separately
 * and must each be complete). Then decide what the threads share: tiles of C
 * are independent, so a parallel loop over them needs no reduction and no
 * atomics -- and if no two threads ever write the same C[i][j], the result
 * stays bit-identical to the serial kernel. Splitting k across threads instead
 * would need one or the other; say in your report why you did not.
 *
 * The harness applies --threads before calling you (do not call
 * omp_set_num_threads yourself); opts->nthreads tells you how many.
 *
 * 1c asks for S(t), E(t) and Amdahl's f from the data: run --threads=1,2,4,...
 * at a fixed N. The serial fraction is what is left when the parallel loop is
 * perfectly split -- if f comes out large, look for work outside it.
 *
 * Built with -fopenmp, WITHOUT -march=native: an intrinsic here fails to
 * compile. That is on purpose.
 *
 * Build and test:
 *   make gemm_omp
 *   ./gemm_omp --n=2048 --threads=8
 */

#include <omp.h>
#include "gemm.h"

void gemm(int n, const float * A, const float * B, float * C,
          const gemm_opts * opts)
{
    /* TODO: your tiled kernel, with the tile loop parallelised. */
    (void)n; (void)A; (void)B; (void)C; (void)opts;
}
