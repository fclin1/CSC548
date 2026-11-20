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
    return "cuda_device";
}

/**
 * Pascal-safe Warp Shuffle Reduction.
 */
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

/**
 * Stage 3 Vectorized float4 Assignment Kernel:
 * 1 thread per 4 points (thread coarsening), float4 vectorized coalesced SoA loads,
 * tiled shared-memory centroids with 4-way register reuse.
 */
static __global__ void assign_opt_float4_kernel(
    const float *points_soa,
    const float *centroids,
    int *assignments,
    int N,
    int D,
    int K,
    int N_pad,
    int k_tile,
    int num_vec)
{
    extern __shared__ float s_centroids[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int base_i = tid * 4;

    float best_dist0 = FLT_MAX, best_dist1 = FLT_MAX, best_dist2 = FLT_MAX, best_dist3 = FLT_MAX;
    int best_c0 = 0, best_c1 = 0, best_c2 = 0, best_c3 = 0;

    for (int c_base = 0; c_base < K; c_base += k_tile) {
        int c_count = (K - c_base < k_tile) ? (K - c_base) : k_tile;
        int total_floats = c_count * D;

        for (int idx = threadIdx.x; idx < total_floats; idx += blockDim.x) {
            s_centroids[idx] = centroids[(size_t)c_base * D + idx];
        }
        __syncthreads();

        if (tid < num_vec && base_i < N) {
            for (int c_local = 0; c_local < c_count; ++c_local) {
                const float *mu = s_centroids + (size_t)c_local * D;
                float dist0 = 0.0f, dist1 = 0.0f, dist2 = 0.0f, dist3 = 0.0f;

                for (int d = 0; d < D; ++d) {
                    const float4 *row_vec = (const float4 *)(points_soa + (size_t)d * N_pad);
                    float4 pt_vec = row_vec[tid];
                    float mu_d = mu[d];

                    float diff0 = pt_vec.x - mu_d;
                    float diff1 = pt_vec.y - mu_d;
                    float diff2 = pt_vec.z - mu_d;
                    float diff3 = pt_vec.w - mu_d;

                    dist0 += diff0 * diff0;
                    dist1 += diff1 * diff1;
                    dist2 += diff2 * diff2;
                    dist3 += diff3 * diff3;
                }

                int c = c_base + c_local;
                if (dist0 < best_dist0) { best_dist0 = dist0; best_c0 = c; }
                if (dist1 < best_dist1) { best_dist1 = dist1; best_c1 = c; }
                if (dist2 < best_dist2) { best_dist2 = dist2; best_c2 = c; }
                if (dist3 < best_dist3) { best_dist3 = dist3; best_c3 = c; }
            }
        }
        __syncthreads();
    }

    if (tid < num_vec) {
        if (base_i + 0 < N) assignments[base_i + 0] = best_c0;
        if (base_i + 1 < N) assignments[base_i + 1] = best_c1;
        if (base_i + 2 < N) assignments[base_i + 2] = best_c2;
        if (base_i + 3 < N) assignments[base_i + 3] = best_c3;
    }
}

/**
 * Path A: Block-Private Shared-Memory Centroid Accumulation Kernel.
 */
static __global__ void accumulate_shmem_kernel(
    const float *points_soa,
    const int *assignments,
    float *new_centroids,
    int *counts,
    int N,
    int D,
    int K,
    int N_pad)
{
    extern __shared__ float s_buf[];
    float *s_sums = s_buf;
    int *s_counts = (int *)(s_buf + (size_t)K * D);

    int total_sum_elements = K * D;
    for (int idx = threadIdx.x; idx < total_sum_elements; idx += blockDim.x) {
        s_sums[idx] = 0.0f;
    }
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        s_counts[c] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int c = assignments[i];
        atomicAdd(&s_counts[c], 1);
        for (int d = 0; d < D; ++d) {
            float val = points_soa[(size_t)d * N_pad + i];
            atomicAdd(&s_sums[(size_t)c * D + d], val);
        }
    }
    __syncthreads();

    /* Flush non-zero block-private sums to global memory */
    for (int idx = threadIdx.x; idx < total_sum_elements; idx += blockDim.x) {
        float val = s_sums[idx];
        if (val != 0.0f) {
            atomicAdd(&new_centroids[idx], val);
        }
    }
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        int cnt = s_counts[c];
        if (cnt > 0) {
            atomicAdd(&counts[c], cnt);
        }
    }
}

/**
 * Path B: Direct Global Atomic Fallback Centroid Accumulation Kernel.
 */
static __global__ void accumulate_global_kernel(
    const float *points_soa,
    const int *assignments,
    float *new_centroids,
    int *counts,
    int N,
    int D,
    int N_pad)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int c = assignments[i];
        atomicAdd(&counts[c], 1);
        for (int d = 0; d < D; ++d) {
            float val = points_soa[(size_t)d * N_pad + i];
            atomicAdd(&new_centroids[(size_t)c * D + d], val);
        }
    }
}

/**
 * Centroid Normalization & Warp-Shuffle Convergence Kernel.
 */
static __global__ void normalize_centroids_kernel(
    float *centroids,
    const float *new_centroids,
    const int *counts,
    float *d_shift_sq,
    int K,
    int D,
    bool check_shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = K * D;
    float local_shift = 0.0f;

    if (idx < total_elements) {
        int c = idx / D;
        int cnt = counts[c];
        if (cnt > 0) {
            float new_val = new_centroids[idx] / (float)cnt;
            if (check_shift) {
                float diff = new_val - centroids[idx];
                local_shift = diff * diff;
            }
            centroids[idx] = new_val;
        }
        /* Empty cluster: centroid is unchanged, shift contribution is 0 */
    }

    if (check_shift) {
        float warp_shift = warp_reduce_sum_f32(local_shift);
        if ((threadIdx.x & 31) == 0 && warp_shift > 0.0f) {
            atomicAdd(d_shift_sq, warp_shift);
        }
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
    int num_vec = N_pad / 4;
    int vec_grid = (num_vec + block_size - 1) / block_size;
    int pt_grid  = (N + block_size - 1) / block_size;

    int k_tile = SHMEM_MAX_FLOATS / D;
    if (k_tile > K) {
        k_tile = K;
    }
    size_t assign_shmem = (size_t)k_tile * D * sizeof(float);

    size_t accum_floats = (size_t)K * D + (size_t)K;
    bool use_shmem_accum = (accum_floats <= ACCUM_SHMEM_MAX_FLOATS);
    size_t accum_shmem_bytes = use_shmem_accum ? (accum_floats * sizeof(float)) : 0;

    memcpy(h_final_centroids, h_init_centroids, (size_t)K * D * sizeof(float));

    /* 1. Setup, Pinned Allocations & Async Stream (timed in setup_ms) */
    double t_setup0 = wall_seconds();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float *h_points_soa = NULL;
    float *h_pinned_shift = NULL;
    CUDA_CHECK(cudaHostAlloc((void **)&h_points_soa, (size_t)D * N_pad * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&h_pinned_shift, sizeof(float), cudaHostAllocDefault));

    convert_aos_to_soa(h_points_aos, N, D, h_points_soa, N_pad);

    float *d_points_soa = NULL;
    float *d_centroids = NULL;
    float *d_new_centroids = NULL;
    int *d_assignments = NULL;
    int *d_counts = NULL;
    float *d_shift_sq = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_points_soa, (size_t)D * N_pad * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_centroids, (size_t)K * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_new_centroids, (size_t)K * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_assignments, (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_counts, (size_t)K * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_shift_sq, sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(d_points_soa, h_points_soa, (size_t)D * N_pad * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_centroids, h_final_centroids, (size_t)K * D * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double t_setup1 = wall_seconds();
    metrics->setup_ms = (t_setup1 - t_setup0) * 1000.0;

    cudaEvent_t ev_a_start, ev_a_stop;
    cudaEvent_t ev_u_start, ev_u_stop;
    CUDA_CHECK(cudaEventCreate(&ev_a_start));
    CUDA_CHECK(cudaEventCreate(&ev_a_stop));
    CUDA_CHECK(cudaEventCreate(&ev_u_start));
    CUDA_CHECK(cudaEventCreate(&ev_u_stop));

    metrics->assign_ms = 0.0;
    metrics->update_ms = 0.0;
    metrics->xfer_ms = 0.0;
    metrics->iters_run = 0;

    int norm_grid = (K * D + 255) / 256;
    bool check_shift = !cfg->fixed_iters;

    int iter = 0;
    for (; iter < cfg->max_iters; ++iter) {
        /* Phase A: float4 Vectorized Assignment Kernel */
        CUDA_CHECK(cudaEventRecord(ev_a_start, stream));
        assign_opt_float4_kernel<<<vec_grid, block_size, assign_shmem, stream>>>(
            d_points_soa, d_centroids, d_assignments, N, D, K, N_pad, k_tile, num_vec);
        CUDA_CHECK(cudaEventRecord(ev_a_stop, stream));

        /* Phase B: Zero Device Buffers + Accumulate + Normalize Kernel */
        CUDA_CHECK(cudaEventRecord(ev_u_start, stream));
        CUDA_CHECK(cudaMemsetAsync(d_new_centroids, 0, (size_t)K * D * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_counts, 0, (size_t)K * sizeof(int), stream));
        if (check_shift) {
            CUDA_CHECK(cudaMemsetAsync(d_shift_sq, 0, sizeof(float), stream));
        }

        if (use_shmem_accum) {
            accumulate_shmem_kernel<<<pt_grid, block_size, accum_shmem_bytes, stream>>>(
                d_points_soa, d_assignments, d_new_centroids, d_counts, N, D, K, N_pad);
        } else {
            accumulate_global_kernel<<<pt_grid, block_size, 0, stream>>>(
                d_points_soa, d_assignments, d_new_centroids, d_counts, N, D, N_pad);
        }

        normalize_centroids_kernel<<<norm_grid, 256, 0, stream>>>(
            d_centroids, d_new_centroids, d_counts, d_shift_sq, K, D, check_shift);
        CUDA_CHECK(cudaEventRecord(ev_u_stop, stream));

        /* Sync events and accumulate timings */
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float a_ms = 0.0f, u_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&a_ms, ev_a_start, ev_a_stop));
        CUDA_CHECK(cudaEventElapsedTime(&u_ms, ev_u_start, ev_u_stop));
        metrics->assign_ms += (double)a_ms;
        metrics->update_ms += (double)u_ms;

        if (check_shift) {
            double tx0 = wall_seconds();
            CUDA_CHECK(cudaMemcpyAsync(h_pinned_shift, d_shift_sq, sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            double tx1 = wall_seconds();
            metrics->xfer_ms += (tx1 - tx0) * 1000.0;

            if (*h_pinned_shift <= cfg->tol * cfg->tol) {
                iter++;
                break;
            }
        }
    }

    metrics->iters_run = (iter == 0) ? 1 : iter;
    metrics->total_compute_ms = metrics->assign_ms + metrics->update_ms + metrics->xfer_ms;
    metrics->ms_per_iter = metrics->total_compute_ms / (double)metrics->iters_run;

    /* Final D2H Copy of final centroids and assignments */
    CUDA_CHECK(cudaMemcpyAsync(h_final_centroids, d_centroids, (size_t)K * D * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_assignments, d_assignments, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* Cleanup */
    CUDA_CHECK(cudaEventDestroy(ev_a_start));
    CUDA_CHECK(cudaEventDestroy(ev_a_stop));
    CUDA_CHECK(cudaEventDestroy(ev_u_start));
    CUDA_CHECK(cudaEventDestroy(ev_u_stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_points_soa));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_new_centroids));
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_shift_sq));

    CUDA_CHECK(cudaFreeHost(h_points_soa));
    CUDA_CHECK(cudaFreeHost(h_pinned_shift));
}
