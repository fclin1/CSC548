#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <cuda_runtime.h>
#include "config.h"
#include "kmeans_types.h"
#include "kmeans_cuda.h"
#include "timer.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_CUDA_ERR); \
    } \
} while (0)

extern "C" const char *kmeans_cuda_impl_name(void) {
    return "cuda_naive";
}

/**
 * Stage 1 Naive Assignment Kernel:
 * 1 thread per point, uncoalesced AoS global memory loads, global memory centroids.
 */
static __global__ void assign_naive_kernel(
    const float *points_aos,
    const float *centroids,
    int *assignments,
    int N,
    int D,
    int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
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
 * Host CPU update function for Stage 1.
 */
static void host_centroid_update(
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
        /* Empty cluster fallback: keep existing centroid */
    }

    if (out_shift_sq) {
        *out_shift_sq = total_shift_sq;
    }
}

void kmeans_cuda_run(
    const float *h_points_aos,
    const float *h_init_centroids,
    float *h_final_centroids,
    int *h_assignments,
    const KMeansConfig *cfg,
    KMeansMetrics *metrics)
{
    int N = cfg->n;
    int D = cfg->d;
    int K = cfg->k;
    int block_size = cfg->block_size;
    int grid_size = (N + block_size - 1) / block_size;

    memcpy(h_final_centroids, h_init_centroids, (size_t)K * D * sizeof(float));

    /* 1. Setup & Device Allocations (timed in setup_ms) */
    double t_setup0 = wall_seconds();

    float *d_points_aos = NULL;
    float *d_centroids = NULL;
    int *d_assignments = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_points_aos, (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_centroids, (size_t)K * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_assignments, (size_t)N * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_points_aos, h_points_aos, (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_final_centroids, (size_t)K * D * sizeof(float), cudaMemcpyHostToDevice));

    double t_setup1 = wall_seconds();
    metrics->setup_ms = (t_setup1 - t_setup0) * 1000.0;

    /* Buffers for host update */
    float *h_sums = (float *)malloc((size_t)K * D * sizeof(float));
    int *h_counts = (int *)malloc((size_t)K * sizeof(int));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    metrics->assign_ms = 0.0;
    metrics->update_ms = 0.0;
    metrics->xfer_ms = 0.0;
    metrics->iters_run = 0;

    int iter = 0;
    for (; iter < cfg->max_iters; ++iter) {
        /* Step A: Assignment Kernel (timed via cudaEvent) */
        CUDA_CHECK(cudaEventRecord(ev_start, 0));
        assign_naive_kernel<<<grid_size, block_size>>>(d_points_aos, d_centroids, d_assignments, N, D, K);
        CUDA_CHECK(cudaEventRecord(ev_stop, 0));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float k_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&k_ms, ev_start, ev_stop));
        metrics->assign_ms += (double)k_ms;

        /* Step B: D2H Transfer of Assignments (timed in xfer_ms) */
        double t_xfer0 = wall_seconds();
        CUDA_CHECK(cudaMemcpy(h_assignments, d_assignments, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
        double t_xfer1 = wall_seconds();
        metrics->xfer_ms += (t_xfer1 - t_xfer0) * 1000.0;

        /* Step C: Host Centroid Update (timed in update_ms) */
        float shift_sq = 0.0f;
        double t_up0 = wall_seconds();
        host_centroid_update(h_points_aos, h_assignments, h_final_centroids, h_sums, h_counts, N, D, K, &shift_sq);
        double t_up1 = wall_seconds();
        metrics->update_ms += (t_up1 - t_up0) * 1000.0;

        /* Step D: H2D Transfer of Updated Centroids (timed in xfer_ms) */
        double t_xfer2 = wall_seconds();
        CUDA_CHECK(cudaMemcpy(d_centroids, h_final_centroids, (size_t)K * D * sizeof(float), cudaMemcpyHostToDevice));
        double t_xfer3 = wall_seconds();
        metrics->xfer_ms += (t_xfer3 - t_xfer2) * 1000.0;

        if (!cfg->fixed_iters && shift_sq <= cfg->tol * cfg->tol) {
            iter++;
            break;
        }
    }

    metrics->iters_run = (iter == 0) ? 1 : iter;
    metrics->total_compute_ms = metrics->assign_ms + metrics->update_ms + metrics->xfer_ms;
    metrics->ms_per_iter = metrics->total_compute_ms / (double)metrics->iters_run;

    /* Cleanup */
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_points_aos));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_assignments));
    free(h_sums);
    free(h_counts);
}
