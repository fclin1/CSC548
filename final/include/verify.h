#ifndef KMEANS_VERIFY_H
#define KMEANS_VERIFY_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "config.h"
#include "kmeans_types.h"
#include "timer.h"
#include "alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define KMEANS_NOINLINE __attribute__((noinline))
#else
#define KMEANS_NOINLINE
#endif

/**
 * Serial CPU Assignment Step (non-inlined for clean profiling in gprof/perf).
 */
KMEANS_NOINLINE void kmeans_cpu_assign(
    const float *points_aos,
    const float *centroids,
    int *assignments,
    int N,
    int D,
    int K)
{
    for (int i = 0; i < N; ++i) {
        const float *pt = points_aos + (size_t)i * D;
        int best_c = 0;
        float best_dist = FLT_MAX;

        for (int c = 0; c < K; ++c) {
            const float *mu = centroids + (size_t)c * D;
            float dist = 0.0f;
            for (int d = 0; d < D; ++d) {
                float diff = pt[d] - mu[d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_c = c;
            }
        }
        assignments[i] = best_c;
    }
}

/**
 * Serial CPU Centroid Update Step with deterministic empty-cluster fallback.
 */
KMEANS_NOINLINE void kmeans_cpu_update(
    const float *points_aos,
    const int *assignments,
    float *centroids,
    float *sums,
    int *counts,
    int N,
    int D,
    int K,
    float *out_shift_sq)
{
    memset(sums, 0, (size_t)K * D * sizeof(float));
    memset(counts, 0, (size_t)K * sizeof(int));

    for (int i = 0; i < N; ++i) {
        int c = assignments[i];
        counts[c]++;
        const float *pt = points_aos + (size_t)i * D;
        float *sum_c = sums + (size_t)c * D;
        for (int d = 0; d < D; ++d) {
            sum_c[d] += pt[d];
        }
    }

    float total_shift_sq = 0.0f;
    for (int c = 0; c < K; ++c) {
        float *mu = centroids + (size_t)c * D;
        if (counts[c] > 0) {
            float inv_n = 1.0f / (float)counts[c];
            const float *sum_c = sums + (size_t)c * D;
            for (int d = 0; d < D; ++d) {
                float new_mu = sum_c[d] * inv_n;
                float diff = new_mu - mu[d];
                total_shift_sq += diff * diff;
                mu[d] = new_mu;
            }
        }
        /* Empty cluster: centroid coordinates remain unchanged, shift = 0 */
    }

    if (out_shift_sq) {
        *out_shift_sq = total_shift_sq;
    }
}

/**
 * Full CPU Reference Lloyd's K-Means execution.
 */
static inline void kmeans_cpu_reference(
    const float *points_aos,
    const float *init_centroids,
    float *final_centroids,
    int *assignments,
    int N,
    int D,
    int K,
    int max_iters,
    bool fixed_iters,
    float tol,
    KMeansMetrics *metrics)
{
    memcpy(final_centroids, init_centroids, (size_t)K * D * sizeof(float));

    float *sums = (float *)malloc((size_t)K * D * sizeof(float));
    int *counts = (int *)malloc((size_t)K * sizeof(int));

    metrics->assign_ms = 0.0;
    metrics->update_ms = 0.0;
    metrics->xfer_ms = 0.0;
    metrics->iters_run = 0;

    int iter = 0;
    for (; iter < max_iters; ++iter) {
        double t0 = wall_seconds();
        kmeans_cpu_assign(points_aos, final_centroids, assignments, N, D, K);
        double t1 = wall_seconds();
        metrics->assign_ms += (t1 - t0) * 1000.0;

        float shift_sq = 0.0f;
        double t2 = wall_seconds();
        kmeans_cpu_update(points_aos, assignments, final_centroids, sums, counts, N, D, K, &shift_sq);
        double t3 = wall_seconds();
        metrics->update_ms += (t3 - t2) * 1000.0;

        if (!fixed_iters && shift_sq <= tol * tol) {
            iter++;
            break;
        }
    }

    metrics->iters_run = (iter == 0) ? 1 : iter;
    metrics->total_compute_ms = metrics->assign_ms + metrics->update_ms;
    metrics->ms_per_iter = metrics->total_compute_ms / (double)metrics->iters_run;

    free(sums);
    free(counts);
}

/**
 * Verify test centroids and labels against reference output.
 */
static inline VerifyResult verify_kmeans_outputs(
    const float *ref_centroids,
    const int *ref_labels,
    const float *test_centroids,
    const int *test_labels,
    int N,
    int D,
    int K)
{
    VerifyResult res;
    memset(&res, 0, sizeof(VerifyResult));

    double sum_diff_sq = 0.0;
    double sum_ref_sq = 0.0;
    double max_abs = 0.0;

    for (size_t i = 0; i < (size_t)K * D; ++i) {
        double diff = (double)test_centroids[i] - (double)ref_centroids[i];
        double abs_diff = fabs(diff);
        if (abs_diff > max_abs) {
            max_abs = abs_diff;
        }
        sum_diff_sq += diff * diff;
        sum_ref_sq += (double)ref_centroids[i] * (double)ref_centroids[i];
    }

    double denom = sqrt(sum_ref_sq);
    if (denom < 1e-12) {
        denom = 1e-12;
    }
    res.centroid_rel_l2 = sqrt(sum_diff_sq) / denom;
    res.centroid_max_abs = max_abs;

    uint64_t match_count = 0;
    for (int i = 0; i < N; ++i) {
        if (test_labels[i] == ref_labels[i]) {
            match_count++;
        }
    }
    res.label_match_pct = (100.0 * (double)match_count) / (double)N;
    res.passed = (res.centroid_rel_l2 <= VERIFY_REL_L2_TOL) &&
                 (res.label_match_pct >= VERIFY_LABEL_MATCH_MIN);

    return res;
}

/**
 * Dump reference centroids and assignments to binary files.
 */
static inline int dump_reference_files(
    const char *prefix,
    const float *centroids,
    const int *labels,
    int N,
    int D,
    int K)
{
    char path_cent[1024];
    char path_lbl[1024];
    snprintf(path_cent, sizeof(path_cent), "%s_centroids.bin", prefix);
    snprintf(path_lbl, sizeof(path_lbl), "%s_labels.bin", prefix);

    FILE *fp_c = fopen(path_cent, "wb");
    if (!fp_c) {
        fprintf(stderr, "Error: Cannot open '%s' for writing\n", path_cent);
        return EXIT_IO_ERR;
    }
    uint64_t hdr_c[2] = { (uint64_t)K, (uint64_t)D };
    fwrite(hdr_c, sizeof(uint64_t), 2, fp_c);
    fwrite(centroids, sizeof(float), (size_t)K * D, fp_c);
    fclose(fp_c);

    FILE *fp_l = fopen(path_lbl, "wb");
    if (!fp_l) {
        fprintf(stderr, "Error: Cannot open '%s' for writing\n", path_lbl);
        return EXIT_IO_ERR;
    }
    uint64_t hdr_l[2] = { (uint64_t)N, 1 };
    fwrite(hdr_l, sizeof(uint64_t), 2, fp_l);
    fwrite(labels, sizeof(int), (size_t)N, fp_l);
    fclose(fp_l);

    printf("Successfully dumped reference files: '%s' and '%s'\n", path_cent, path_lbl);
    return EXIT_OK;
}

/**
 * Load reference centroids and assignments from binary files.
 */
static inline int load_reference_files(
    const char *prefix,
    float **centroids,
    int **labels,
    int N,
    int D,
    int K)
{
    char path_cent[1024];
    char path_lbl[1024];
    snprintf(path_cent, sizeof(path_cent), "%s_centroids.bin", prefix);
    snprintf(path_lbl, sizeof(path_lbl), "%s_labels.bin", prefix);

    FILE *fp_c = fopen(path_cent, "rb");
    if (!fp_c) {
        fprintf(stderr, "Error: Cannot open reference file '%s'\n", path_cent);
        return EXIT_IO_ERR;
    }
    uint64_t hdr_c[2] = { 0, 0 };
    if (fread(hdr_c, sizeof(uint64_t), 2, fp_c) != 2 ||
        (int)hdr_c[0] != K || (int)hdr_c[1] != D) {
        fprintf(stderr, "Error: Reference centroid dimension mismatch in '%s'\n", path_cent);
        fclose(fp_c);
        return EXIT_DATA_ERR;
    }
    float *cent = (float *)malloc((size_t)K * D * sizeof(float));
    if (fread(cent, sizeof(float), (size_t)K * D, fp_c) != (size_t)K * D) {
        fprintf(stderr, "Error: Truncated reference centroid data in '%s'\n", path_cent);
        free(cent);
        fclose(fp_c);
        return EXIT_IO_ERR;
    }
    fclose(fp_c);

    FILE *fp_l = fopen(path_lbl, "rb");
    if (!fp_l) {
        fprintf(stderr, "Error: Cannot open reference label file '%s'\n", path_lbl);
        free(cent);
        return EXIT_IO_ERR;
    }
    uint64_t hdr_l[2] = { 0, 0 };
    if (fread(hdr_l, sizeof(uint64_t), 2, fp_l) != 2 ||
        (int)hdr_l[0] != N || hdr_l[1] != 1) {
        fprintf(stderr, "Error: Reference label size mismatch in '%s'\n", path_lbl);
        free(cent);
        fclose(fp_l);
        return EXIT_DATA_ERR;
    }
    int *lbl = (int *)malloc((size_t)N * sizeof(int));
    if (fread(lbl, sizeof(int), (size_t)N, fp_l) != (size_t)N) {
        fprintf(stderr, "Error: Truncated reference label data in '%s'\n", path_lbl);
        free(cent);
        free(lbl);
        fclose(fp_l);
        return EXIT_IO_ERR;
    }
    fclose(fp_l);

    *centroids = cent;
    *labels = lbl;
    return EXIT_OK;
}

/**
 * Standardized GFLOP/s and GB/s calculations.
 */
static inline void compute_derived_metrics(KMeansMetrics *m, int N, int D, int K) {
    if (m->total_compute_ms <= 0.0 || m->iters_run <= 0) {
        m->gflops = 0.0;
        m->gbytes_s = 0.0;
        return;
    }
    double flops_iter = (double)N * (double)D * (3.0 * (double)K + 1.0) + (double)K * (double)D;
    double bytes_iter = 4.0 * ((double)N * (double)D + (double)N + (double)K * (double)D);

    m->gflops = (flops_iter * (double)m->iters_run) / (m->total_compute_ms * 1e6);
    m->gbytes_s = (bytes_iter * (double)m->iters_run) / (m->total_compute_ms * 1e6);
}

static inline int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static inline double median_double(double *arr, int n) {
    if (n <= 0) return 0.0;
    qsort(arr, (size_t)n, sizeof(double), compare_doubles);
    if (n % 2 == 1) {
        return arr[n / 2];
    } else {
        return 0.5 * (arr[n / 2 - 1] + arr[n / 2]);
    }
}

static inline void print_csv_header(FILE *fp) {
    fprintf(fp, "impl,dataset,N,D,K,iters_run,block_size,trial_count,io_ms,setup_ms,assign_ms,update_ms,xfer_ms,total_compute_ms,ms_per_iter,gflops,gbytes_s,verify_status,centroid_rel_l2,centroid_max_abs,label_match_pct\n");
}

static inline int append_csv_row(
    const char *csv_path,
    const char *impl_name,
    const KMeansConfig *cfg,
    const KMeansMetrics *m,
    const VerifyResult *v)
{
    FILE *fp = fopen(csv_path, "a+");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open CSV file '%s' for writing\n", csv_path);
        return EXIT_IO_ERR;
    }

    /* Check if file is empty to print header */
    fseek(fp, 0, SEEK_END);
    if (ftell(fp) == 0) {
        print_csv_header(fp);
    }

    const char *dataset_name = cfg->synthetic ? "synthetic" : cfg->file_path;
    const char *verify_str = "NOT_CHECKED";
    double rel_l2 = 0.0;
    double max_abs = 0.0;
    double label_match = 100.0;

    if (v) {
        verify_str = v->passed ? "PASS" : "FAIL";
        rel_l2 = v->centroid_rel_l2;
        max_abs = v->centroid_max_abs;
        label_match = v->label_match_pct;
    }

    fprintf(fp, "%s,%s,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%.6e,%.6e,%.4f\n",
        impl_name,
        dataset_name,
        cfg->n,
        cfg->d,
        cfg->k,
        m->iters_run,
        cfg->block_size,
        cfg->trials,
        m->io_ms,
        m->setup_ms,
        m->assign_ms,
        m->update_ms,
        m->xfer_ms,
        m->total_compute_ms,
        m->ms_per_iter,
        m->gflops,
        m->gbytes_s,
        verify_str,
        rel_l2,
        max_abs,
        label_match);

    fclose(fp);
    return EXIT_OK;
}

#ifdef __cplusplus
}
#endif

#endif /* KMEANS_VERIFY_H */
