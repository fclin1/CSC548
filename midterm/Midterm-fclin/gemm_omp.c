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
    const int tile = opts->tile;
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += tile) {
        const int imax = (ii + tile < n) ? ii + tile : n;
        for (int jj = 0; jj < n; jj += tile) {
            const int jmax = (jj + tile < n) ? jj + tile : n;
            for (int kk = 0; kk < n; kk += tile) {
                const int kmax = (kk + tile < n) ? kk + tile : n;
                for (int i = ii; i < imax; i++) {
                    const float * restrict Ai = A + (size_t)i * n;
                    float * restrict Ci = C + (size_t)i * n;
                    for (int k = kk; k < kmax; k++) {
                        const float a = Ai[k];
                        const float * restrict Bk = B + (size_t)k * n;
                        for (int j = jj; j < jmax; j++) {
                            Ci[j] += a * Bk[j];
                        }
                    }
                }
            }
        }
    }
}

