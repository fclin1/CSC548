/* Midterm 1b: loop reordering + cache blocking. Single thread.
 *
 * ONE OF THE FOUR FILES YOU SUBMIT. Read include/gemm.h for the contract.
 *
 * Two changes to gemm_naive.c, and your report should show what each buys:
 *   1. ikj order, so the inner loop walks a row of B and a row of C with
 *      stride 1 instead of a column of B with stride n.
 *   2. Blocking: work on tile x tile sub-blocks of C (and a tile-deep slice of
 *      k) so the pieces of A, B and C in flight stay in cache while they are
 *      reused. opts->tile is the knob; sweep it.
 * n is a multiple of 32, not of tile -- handle the partial last tile.
 *
 * Built WITHOUT -fopenmp and WITHOUT -march=native: this stage is what one
 * core does with the memory hierarchy, nothing else. A `#pragma omp` here is
 * ignored; an #include <omp.h> or an intrinsic fails to compile.
 *
 * Build and test:
 *   make gemm_tiled
 *   ./gemm_tiled --n=1024 --tile=64
 * The harness prints PASS or FAIL against its reference. A kernel that adds
 * each C[i][j]'s k terms in increasing k, straight into C, reports exactly
 * 0.0; a register accumulator added to C at the end reports ~1e-7. Both pass.
 */

#include "gemm.h"

void gemm(int n, const float * A, const float * B, float * C,
          const gemm_opts * opts)
{
    const int tile = opts->tile;
    for (int ii = 0; ii < n; ii += tile) {
        const int imax = (ii + tile < n) ? ii + tile : n;
        for (int kk = 0; kk < n; kk += tile) {
            const int kmax = (kk + tile < n) ? kk + tile : n;
            for (int jj = 0; jj < n; jj += tile) {
                const int jmax = (jj + tile < n) ? jj + tile : n;
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

