#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"
#include "kmeans_types.h"
#include "cmdline.h"
#include "input.h"
#include "verify.h"
#include "timer.h"
#include "alloc.h"

int main(int argc, char **argv) {
    KMeansConfig cfg;
    int rc = parse_args(argc, argv, &cfg);
    if (rc != EXIT_OK) {
        return rc;
    }

    if (cfg.csv_header) {
        print_csv_header(stdout);
        return EXIT_OK;
    }

    /* 1. Ingest Data */
    float *points_aos = NULL;
    int N = 0;
    int D = 0;
    double t_io0 = wall_seconds();

    if (cfg.synthetic) {
        N = cfg.n;
        D = cfg.d;
        rc = generate_synthetic_points(N, D, cfg.k, cfg.seed, &points_aos);
        if (rc != EXIT_OK) return rc;
    } else {
        rc = load_binary_dataset(cfg.file_path, cfg.n, &points_aos, &N, &D);
        if (rc != EXIT_OK) return rc;
        cfg.n = N;
        cfg.d = D;
    }
    double t_io1 = wall_seconds();
    double io_ms = (t_io1 - t_io0) * 1000.0;

    /* 2. Validate Dataset */
    rc = validate_dataset(points_aos, N, D, cfg.k);
    if (rc != EXIT_OK) {
        aligned_free_64(points_aos);
        return rc;
    }

    /* 3. Setup Initial Centroids */
    double t_setup0 = wall_seconds();
    float *init_centroids = (float *)aligned_alloc_64((size_t)cfg.k * D * sizeof(float));
    float *final_centroids = (float *)aligned_alloc_64((size_t)cfg.k * D * sizeof(float));
    int *assignments = (int *)malloc((size_t)N * sizeof(int));

    if (!init_centroids || !final_centroids || !assignments) {
        fprintf(stderr, "Error: Memory allocation failed during setup\n");
        aligned_free_64(points_aos);
        if (init_centroids) aligned_free_64(init_centroids);
        if (final_centroids) aligned_free_64(final_centroids);
        if (assignments) free(assignments);
        return EXIT_IO_ERR;
    }

    init_centroids_deterministic(points_aos, N, D, cfg.k, cfg.seed, init_centroids);
    double t_setup1 = wall_seconds();
    double setup_ms = (t_setup1 - t_setup0) * 1000.0;

    /* 4. Warmup Runs */
    KMeansMetrics dummy_m;
    for (int w = 0; w < cfg.warmup; ++w) {
        kmeans_cpu_reference(points_aos, init_centroids, final_centroids, assignments,
                             N, D, cfg.k, cfg.max_iters, cfg.fixed_iters, cfg.tol, &dummy_m);
    }

    /* 5. Timed Trials */
    int trials = (cfg.trials > 0) ? cfg.trials : 1;
    double *trial_assign = (double *)malloc((size_t)trials * sizeof(double));
    double *trial_update = (double *)malloc((size_t)trials * sizeof(double));
    double *trial_total  = (double *)malloc((size_t)trials * sizeof(double));
    KMeansMetrics last_m;

    for (int t = 0; t < trials; ++t) {
        kmeans_cpu_reference(points_aos, init_centroids, final_centroids, assignments,
                             N, D, cfg.k, cfg.max_iters, cfg.fixed_iters, cfg.tol, &last_m);
        trial_assign[t] = last_m.assign_ms;
        trial_update[t] = last_m.update_ms;
        trial_total[t]  = last_m.total_compute_ms;
    }

    KMeansMetrics med_m;
    med_m.iters_run = last_m.iters_run;
    med_m.io_ms = io_ms;
    med_m.setup_ms = setup_ms;
    med_m.assign_ms = median_double(trial_assign, trials);
    med_m.update_ms = median_double(trial_update, trials);
    med_m.xfer_ms = 0.0;
    med_m.total_compute_ms = median_double(trial_total, trials);
    med_m.ms_per_iter = med_m.total_compute_ms / (double)med_m.iters_run;

    compute_derived_metrics(&med_m, N, D, cfg.k);

    free(trial_assign);
    free(trial_update);
    free(trial_total);

    /* 6. Dump Reference Files if requested */
    if (cfg.dump_ref_prefix[0] != '\0') {
        dump_reference_files(cfg.dump_ref_prefix, final_centroids, assignments, N, D, cfg.k);
    }

    /* 7. Check Reference Files or Self-Verify */
    VerifyResult v;
    VerifyResult *v_ptr = NULL;

    if (cfg.check_ref_prefix[0] != '\0') {
        float *ref_cent = NULL;
        int *ref_lbl = NULL;
        rc = load_reference_files(cfg.check_ref_prefix, &ref_cent, &ref_lbl, N, D, cfg.k);
        if (rc == EXIT_OK) {
            v = verify_kmeans_outputs(ref_cent, ref_lbl, final_centroids, assignments, N, D, cfg.k);
            v_ptr = &v;
            free(ref_cent);
            free(ref_lbl);
        }
    } else if (cfg.verify) {
        /* Self parity check for baseline */
        v = verify_kmeans_outputs(final_centroids, assignments, final_centroids, assignments, N, D, cfg.k);
        v_ptr = &v;
    }

    /* 8. Print Summary Output */
    printf("================================================================================\n");
    printf("K-Means Serial Baseline (kmeans_seq)\n");
    printf("Dataset: %s | N: %d | D: %d | K: %d | Iters: %d (Fixed: %s)\n",
           cfg.synthetic ? "synthetic" : cfg.file_path, N, D, cfg.k, med_m.iters_run, cfg.fixed_iters ? "yes" : "no");
    printf("Trials: %d (Warmup: %d)\n", trials, cfg.warmup);
    printf("--------------------------------------------------------------------------------\n");
    printf("Timing Breakdown:\n");
    printf("  I/O Time:           %10.3f ms\n", med_m.io_ms);
    printf("  Setup Time:         %10.3f ms\n", med_m.setup_ms);
    printf("  Assignment Time:    %10.3f ms (%.1f%% of compute)\n",
           med_m.assign_ms, (med_m.assign_ms / med_m.total_compute_ms) * 100.0);
    printf("  Update Time:        %10.3f ms (%.1f%% of compute)\n",
           med_m.update_ms, (med_m.update_ms / med_m.total_compute_ms) * 100.0);
    printf("  Total Compute:      %10.3f ms\n", med_m.total_compute_ms);
    printf("  Per-Iteration:      %10.3f ms\n", med_m.ms_per_iter);
    printf("Throughput:\n");
    printf("  Compute Rate:       %10.3f GFLOP/s\n", med_m.gflops);
    printf("  Effective Bandwidth:%10.3f GB/s\n", med_m.gbytes_s);

    if (v_ptr) {
        printf("Verification:\n");
        printf("  Status:             %s\n", v_ptr->passed ? "PASS" : "FAIL");
        printf("  Centroid Rel L2:    %10.6e (tol: <= %.1e)\n", v_ptr->centroid_rel_l2, VERIFY_REL_L2_TOL);
        printf("  Centroid Max Abs:   %10.6e\n", v_ptr->centroid_max_abs);
        printf("  Label Match:        %10.4f%% (min: >= %.2f%%)\n", v_ptr->label_match_pct, VERIFY_LABEL_MATCH_MIN);
    }
    printf("================================================================================\n");

    /* 9. Append CSV Row */
    if (cfg.csv_path[0] != '\0') {
        cfg.block_size = 0; /* 0 for CPU serial */
        append_csv_row(cfg.csv_path, "kmeans_seq", &cfg, &med_m, v_ptr);
    }

    /* Cleanup */
    aligned_free_64(points_aos);
    aligned_free_64(init_centroids);
    aligned_free_64(final_centroids);
    free(assignments);

    if (v_ptr && !v_ptr->passed) {
        return EXIT_VERIFY_ERR;
    }

    return EXIT_OK;
}
