#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <cuda_runtime.h>
#include "config.h"
#include "kmeans_types.h"
#include "kmeans_cuda.h"
#include "input.h"
#include "timer.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_CUDA_ERR); \
    } \
} while (0)

const char *kmeans_cuda_impl_name(void) {
    return "cuda_shmem";
}

/**
 * Stage 2 Assignment Kernel:
 * Padded SoA coalesced global memory loads + tiled shared-memory centroids broadcast.
 */
static __global__ void assign_shmem_tiled_kernel(
    const float *points_soa,
    const float *centroids,
    int *assignments,
    int N,
    int D,
    int K,
    int N_pad,
    int k_tile)
{
    extern __shared__ float s_centroids[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int best_c = 0;
    float best_dist = FLT_MAX;

    for (int c_base = 0; c_base < K; c_base += k_tile) {
        int c_count = (K - c_base < k_tile) ? (K - c_base) : k_tile;
        int total_floats = c_count * D;

        /* Cooperative loading of centroid tile into shared memory */
        for (int idx = threadIdx.x; idx < total_floats; idx += blockDim.x) {
            s_centroids[idx] = centroids[(size_t)c_base * D + idx];
        }
        __syncthreads();

        if (i < N) {
            for (int c_local = 0; c_local < c_count; ++c_local) {
                const float *mu = s_centroids + (size_t)c_local * D;
                float dist = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float pt_val = points_soa[(size_t)d * N_pad + i];
                    float diff = pt_val - mu[d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c_base + c_local;
                }
            }
        }
        __syncthreads();
    }

    if (i < N) {
        assignments[i] = best_c;
    }
}

/**
 * Host CPU update function for Stage 2.
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
        /* Empty cluster fallback */
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
    int N_pad = compute_padded_n(N);
    int grid_size = (N + block_size - 1) / block_size;

    int k_tile = SHMEM_MAX_FLOATS / D;
    if (k_tile > K) {
        k_tile = K;
    }
    size_t shmem_bytes = (size_t)k_tile * D * sizeof(float);

    memcpy(h_final_centroids, h_init_centroids, (size_t)K * D * sizeof(float));

    /* 1. Setup, SoA Transpose & Device Allocation (timed in setup_ms) */
    double t_setup0 = wall_seconds();

    float *h_points_soa = NULL;
    CUDA_CHECK(cudaHostAlloc((void **)&h_points_soa, (size_t)D * N_pad * sizeof(float), cudaHostAllocDefault));
    convert_aos_to_soa(h_points_aos, N, D, h_points_soa, N_pad);

    float *d_points_soa = NULL;
    float *d_centroids = NULL;
    int *d_assignments = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_points_soa, (size_t)D * N_pad * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_centroids, (size_t)K * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_assignments, (size_t)N * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_points_soa, h_points_soa, (size_t)D * N_pad * sizeof(float), cudaMemcpyHostToDevice));
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
        assign_shmem_tiled_kernel<<<grid_size, block_size, shmem_bytes>>>(
            d_points_soa, d_centroids, d_assignments, N, D, K, N_pad, k_tile);
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
    CUDA_CHECK(cudaFree(d_points_soa));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFreeHost(h_points_soa));
    free(h_sums);
    free(h_counts);
}
