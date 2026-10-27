/* HW3 CUDA harness -- PROVIDED, DO NOT MODIFY.
 *
 * Reads a MatrixMarket matrix, converts it to CSR, hands it to your kernel,
 * checks the result against the sequential CSR baseline, and reports timing
 * both with and without the PCIe transfers.
 *
 * This file is host C and contains no CUDA: every cudaMalloc, cudaMemcpy and
 * launch is in your .cu (Task 1).
 *
 *   ./spmv_cuda_scalar matrix.mtx [options]
 *
 *   --block=N       threads per block (default 256). Sweep for the GPU behavior report.
 *   --iters=N       fix the timed iteration count (default: aim for about
 *                   TIME_LIMIT seconds)
 *   --no-check      skip the correctness check
 *   --csv           one machine-readable result line instead of the report
 *   --csv-header    print the CSV column names and exit
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"
#include "formats_ell.h"
#include "alloc.h"
#include "spmv_cuda.h"

/* Relative L2 allowed against the baseline.
 *
 * Do not expect 0.0 here. nvcc contracts a*b+c into a single FMA by default,
 * and a warp-per-row kernel reduces 32 lanes in a different order than the
 * scalar sum -- both move the last bits. Drift at this tolerance is normal; an
 * error near it is a real bug. */
#define CHECK_TOL 1e-5

#define DEFAULT_BLOCK 256

static const char * CSV_HEADER =
    "matrix,rows,cols,nnz,block,grid,threads_per_row,iters,"
    "ms_kernel,ms_with_transfer,transfer_pct,gflops,gbytes,setup_ms,"
    "max_row_len,ell_slots,pad_pct,check,rel_err";

static void usage(char** argv)
{
    printf("Usage: %s my_matrix.mtx [--block=N] [--iters=N] [--no-check] [--csv]\n",
           argv[0]);
    printf("  --block=N       threads per block (default %d)\n", DEFAULT_BLOCK);
}

int main(int argc, char** argv)
{
    if (get_arg(argc, argv, "csv-header") != NULL) {
        printf("%s\n", CSV_HEADER);
        return 0;
    }
    if (argc < 2 || get_arg(argc, argv, "help") != NULL) {
        usage(argv);
        return argc < 2 ? -1 : 0;
    }

    const char * mm_filename = argv[1];
    int fixed_iters = get_argval_int(argc, argv, "iters", 0);
    int do_check    = (get_arg(argc, argv, "no-check") == NULL);
    int csv         = (get_arg(argc, argv, "csv") != NULL);

    cuda_opts opts;
    opts.block_size = get_argval_int(argc, argv, "block", DEFAULT_BLOCK);
    if (opts.block_size <= 0 || opts.block_size > 1024 || opts.block_size % 32 != 0) {
        fprintf(stderr, "bad --block=%d (a multiple of 32, at most 1024)\n",
                opts.block_size);
        return -1;
    }

    /* ---- load (not timed) ---- */
    coo_matrix coo;
    read_coo_matrix(&coo, mm_filename);

    float * x     = (float*)alloc_array(coo.num_cols, sizeof(float));
    float * y     = (float*)alloc_array(coo.num_rows, sizeof(float));
    float * y_ref = (float*)alloc_array(coo.num_rows, sizeof(float));
    fill_problem(&coo, x, 13);

    csr_matrix A;
    coo_to_csr(&coo, &A);
    delete_coo_matrix(&coo);

    /* ---- your setup: device allocation, upload of A, launch config ---- */
    void * ctx = NULL;
    double t0 = wall_seconds();
    int setup_rc = spmv_cuda_setup(&A, &opts, &ctx);
    double setup_ms = (wall_seconds() - t0) * 1000.0;
    if (setup_rc != 0) {
        fprintf(stderr, "spmv_cuda_setup failed (rc=%d)\n", setup_rc);
        return 1;
    }

    /* x goes up once; the kernel-only loop below reuses it on the device. */
    spmv_cuda_h2d(ctx, x);

    /* ---- correctness against the sequential CSR baseline ---- */
    int passed = 1;
    double rel_err = 0.0, max_abs = 0.0;
    if (do_check) {
        /* zero first, so a submission that never copies back fails
         * deterministically instead of comparing uninitialised memory */
        for (int i = 0; i < A.num_rows; i++) y[i] = 0.0f;
        spmv_cuda_kernel(ctx);
        spmv_cuda_d2h(ctx, y);
        csr_spmv(&A, x, y_ref);

        double num = 0.0, den = 0.0;
        for (int i = 0; i < A.num_rows; i++) {
            double d = (double)y[i] - (double)y_ref[i];
            num += d * d;
            den += (double)y_ref[i] * (double)y_ref[i];
            if (fabs(d) > max_abs) max_abs = fabs(d);
        }
        rel_err = (den > 0.0) ? sqrt(num / den) : sqrt(num);
        passed  = (rel_err < CHECK_TOL) && !isnan(rel_err);
    }

    /* ---- iteration count, from a warmed-up kernel ---- */
    spmv_cuda_kernel(ctx);                       //warm up: context, caches, clocks
    t0 = wall_seconds();
    spmv_cuda_kernel(ctx);
    double est = wall_seconds() - t0;
    int num_iterations = fixed_iters > 0 ? fixed_iters : pick_iterations(est);

    /* ---- timed loop 1: the kernel alone, A/x/y resident on the device ----
     * This is the number the roofline is about: it is the only one that
     * reflects device bandwidth rather than PCIe. */
    t0 = wall_seconds();
    for (int j = 0; j < num_iterations; j++)
        spmv_cuda_kernel(ctx);
    double sec_kernel = (wall_seconds() - t0) / (double)num_iterations;

    /* ---- timed loop 2: upload x, run, download y ----
     * What one SpMV costs if the vectors have to cross PCIe every time, which
     * is what a lone SpMV in a host program really costs. An iterative solver
     * keeps them resident, which is why both numbers are worth reporting. */
    t0 = wall_seconds();
    for (int j = 0; j < num_iterations; j++) {
        spmv_cuda_h2d(ctx, x);
        spmv_cuda_kernel(ctx);
        spmv_cuda_d2h(ctx, y);
    }
    double sec_total = (wall_seconds() - t0) / (double)num_iterations;

    /* ---- what the launch looked like ---- */
    cuda_stats st;
    memset(&st, 0, sizeof st);
    spmv_cuda_stats(ctx, &st);

    int is_ell   = (st.max_row_len > 0);
    int K_ref    = csr_max_row_len(&A);
    int k_ok     = !is_ell || (st.max_row_len == K_ref);
    long long slots_ref = (long long)A.num_rows * (long long)K_ref;
    double pad_pct = (is_ell && slots_ref > 0)
                   ? 100.0 * (double)(slots_ref - A.num_nonzeros) / (double)slots_ref : 0.0;

    double GFLOPs = (sec_kernel == 0) ? 0 :
                    (2.0 * (double)A.num_nonzeros / sec_kernel) / 1e9;
    size_t traffic = is_ell ? bytes_per_ell_spmv_k(A.num_rows, st.max_row_len)
                            : bytes_per_csr_spmv(&A);
    double GBYTEs = (sec_kernel == 0) ? 0 :
                    ((double)traffic / sec_kernel) / 1e9;
    double transfer_pct = (sec_total > 0.0)
                        ? 100.0 * (sec_total - sec_kernel) / sec_total : 0.0;

    const char * verdict = !do_check ? "SKIP" : ((passed && k_ok) ? "PASS" : "FAIL");

    if (csv) {
        printf("%s,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.2f,%.4f,%.4f,%.4f,"
               "%d,%lld,%.2f,%s,%.3e\n",
               mm_filename, A.num_rows, A.num_cols, A.num_nonzeros,
               st.block_size, st.grid_size, st.threads_per_row, num_iterations,
               sec_kernel * 1000.0, sec_total * 1000.0, transfer_pct,
               GFLOPs, GBYTEs, setup_ms,
               st.max_row_len, st.ell_slots, pad_pct, verdict, rel_err);
    } else {
        size_t csr_bytes = storage_bytes_csr(&A);
        printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n",
               mm_filename, A.num_rows, A.num_cols, A.num_nonzeros);
        printf("launch: %d blocks x %d threads, %d thread(s) per row\n",
               st.grid_size, st.block_size, st.threads_per_row);
        if (is_ell) {
            size_t ell_bytes = storage_bytes_ell_k(A.num_rows, st.max_row_len);
            printf("storage: CSR %.2f MB, ELL %.2f MB (ELL/CSR = %.2f)\n",
                   csr_bytes / 1e6, ell_bytes / 1e6,
                   (double)ell_bytes / (double)csr_bytes);
            printf("ELL: K=%d, %lld slots, %lld of them padding (%.1f%%)\n",
                   st.max_row_len, st.ell_slots,
                   st.ell_slots - (long long)A.num_nonzeros, pad_pct);
            if (!k_ok)
                printf("  !! K should be %d for this matrix -- check your conversion\n",
                       K_ref);
        } else {
            printf("storage: CSR %.2f MB\n", csr_bytes / 1e6);
        }
        printf("setup (device alloc + upload of A): %.4f ms\n", setup_ms);
        if (do_check)
            printf("correctness: %s (relative L2 error %.3e, max |diff| %.3e, tolerance %.0e)\n",
                   verdict, rel_err, max_abs, (double)CHECK_TOL);
        else
            printf("correctness: skipped\n");
        printf("\tPerforming %d iterations\n", num_iterations);
        printf("\tkernel only:      %8.4f ms ( %6.2f GFLOP/s %7.1f GB/s)\n",
               sec_kernel * 1000.0, GFLOPs, GBYTEs);
        printf("\twith H2D + D2H:   %8.4f ms (transfer is %.1f%% of it)\n",
               sec_total * 1000.0, transfer_pct);
    }
    fflush(stdout);

    spmv_cuda_teardown(ctx);
    delete_csr_matrix(&A);
    free(x); free(y); free(y_ref);

    return (do_check && !(passed && k_ok)) ? 1 : 0;
}
