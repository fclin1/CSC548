#ifndef KMEANS_INPUT_H
#define KMEANS_INPUT_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 64-bit SplitMix64 PRNG for bit-exact cross-platform determinism */
typedef struct {
    uint64_t state;
} RngState;

static inline uint64_t rng_next_u64(RngState *rng) {
    uint64_t z = (rng->state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* Uniform double strictly in (0, 1] for safe log() in Box-Muller */
static inline double rng_uniform_open0(RngState *rng) {
    return ((double)(rng_next_u64(rng) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

/**
 * Compute 128-float padded stride N_pad = ceil(N / 128) * 128.
 */
static inline int compute_padded_n(int N) {
    return ((N + 127) / 128) * 128;
}

/**
 * Transpose row-major AoS to column-major padded SoA.
 * Zero-fills padding elements [N, N_pad - 1].
 */
static inline void convert_aos_to_soa(const float *aos, int N, int D, float *soa, int N_pad) {
    for (int d = 0; d < D; ++d) {
        float *soa_row = soa + (size_t)d * N_pad;
        for (int i = 0; i < N; ++i) {
            soa_row[i] = aos[(size_t)i * D + d];
        }
        for (int i = N; i < N_pad; ++i) {
            soa_row[i] = 0.0f;
        }
    }
}

/**
 * Validate input coordinates and dataset parameters.
 */
static inline int validate_dataset(const float *aos, int N, int D, int K) {
    if (N <= 0 || D <= 0 || K <= 0) {
        fprintf(stderr, "Data validation error: Non-positive parameter (N=%d, D=%d, K=%d)\n", N, D, K);
        return EXIT_CLI_ERR;
    }
    if (N < K) {
        fprintf(stderr, "Data validation error: N (%d) is less than K (%d)\n", N, K);
        return EXIT_DATA_ERR;
    }
    if (D > MAX_SUPPORTED_D) {
        fprintf(stderr, "Data validation error: D (%d) exceeds limit (%d)\n", D, MAX_SUPPORTED_D);
        return EXIT_DATA_ERR;
    }
    for (size_t i = 0; i < (size_t)N; ++i) {
        for (size_t d = 0; d < (size_t)D; ++d) {
            float val = aos[i * D + d];
            if (!isfinite(val)) {
                fprintf(stderr, "Data validation error: Non-finite float at point %zu, dim %zu: %f\n", i, d, (double)val);
                return EXIT_DATA_ERR;
            }
        }
    }
    return EXIT_OK;
}

/**
 * Deterministic synthetic Gaussian mixture generator.
 */
static inline int generate_synthetic_points(int N, int D, int K, uint32_t seed, float **out_aos) {
    float *aos = (float *)aligned_alloc_64((size_t)N * D * sizeof(float));
    if (!aos) {
        fprintf(stderr, "Memory allocation error for synthetic points (%zu bytes)\n", (size_t)N * D * sizeof(float));
        return EXIT_IO_ERR;
    }

    RngState rng = { (uint64_t)seed };

    /* Generate K ground-truth centers in [-10.0, 10.0]^D */
    float *centers = (float *)malloc((size_t)K * D * sizeof(float));
    if (!centers) {
        aligned_free_64(aos);
        return EXIT_IO_ERR;
    }
    for (int c = 0; c < K; ++c) {
        for (int d = 0; d < D; ++d) {
            double u = rng_uniform_open0(&rng);
            centers[c * D + d] = (float)(-10.0 + 20.0 * u);
        }
    }

    /* Generate N points with Box-Muller Gaussian noise around centers */
    for (int i = 0; i < N; ++i) {
        int c = i % K;
        for (int d = 0; d < D; d += 2) {
            double u1 = rng_uniform_open0(&rng);
            double u2 = rng_uniform_open0(&rng);
            double r = sqrt(-2.0 * log(u1));
            double z0 = r * cos(2.0 * M_PI * u2);
            double z1 = r * sin(2.0 * M_PI * u2);

            aos[(size_t)i * D + d] = centers[c * D + d] + (float)z0;
            if (d + 1 < D) {
                aos[(size_t)i * D + d + 1] = centers[c * D + d + 1] + (float)z1;
            }
        }
    }

    free(centers);
    *out_aos = aos;
    return EXIT_OK;
}

/**
 * Deterministic centroid initialization via partial Fisher-Yates shuffle.
 */
static inline void init_centroids_deterministic(const float *aos, int N, int D, int K, uint32_t seed, float *out_centroids) {
    RngState rng = { (uint64_t)seed + 1337ULL };
    int *indices = (int *)malloc((size_t)N * sizeof(int));
    if (!indices) {
        /* Fallback: direct prefix copy if malloc fails */
        memcpy(out_centroids, aos, (size_t)K * D * sizeof(float));
        return;
    }
    for (int i = 0; i < N; ++i) {
        indices[i] = i;
    }
    for (int c = 0; c < K; ++c) {
        uint64_t r = rng_next_u64(&rng) % (uint64_t)(N - c);
        int j = c + (int)r;
        int tmp = indices[c];
        indices[c] = indices[j];
        indices[j] = tmp;

        memcpy(out_centroids + (size_t)c * D, aos + (size_t)indices[c] * D, (size_t)D * sizeof(float));
    }
    free(indices);
}

/**
 * Load binary dataset (.bin) with little-endian 16-byte header:
 * uint64_t N, uint64_t D, followed by N*D floats.
 */
static inline int load_binary_dataset(const char *path, int n_limit, float **out_aos, int *out_N, int *out_D) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open binary dataset '%s'\n", path);
        return EXIT_IO_ERR;
    }

    uint64_t file_n = 0, file_d = 0;
    if (fread(&file_n, sizeof(uint64_t), 1, fp) != 1 ||
        fread(&file_d, sizeof(uint64_t), 1, fp) != 1) {
        fprintf(stderr, "Error: Truncated header in binary dataset '%s'\n", path);
        fclose(fp);
        return EXIT_IO_ERR;
    }

    int N = (int)file_n;
    int D = (int)file_d;
    if (n_limit > 0 && n_limit < N) {
        N = n_limit;
    }

    size_t total_floats = (size_t)N * D;
    float *aos = (float *)aligned_alloc_64(total_floats * sizeof(float));
    if (!aos) {
        fprintf(stderr, "Error: Memory allocation failed for dataset (%zu bytes)\n", total_floats * sizeof(float));
        fclose(fp);
        return EXIT_IO_ERR;
    }

    size_t read_floats = fread(aos, sizeof(float), total_floats, fp);
    fclose(fp);

    if (read_floats != total_floats) {
        fprintf(stderr, "Error: Truncated binary data in '%s': read %zu of %zu floats\n", path, read_floats, total_floats);
        aligned_free_64(aos);
        return EXIT_IO_ERR;
    }

    *out_aos = aos;
    *out_N = N;
    *out_D = D;
    return EXIT_OK;
}

#ifdef __cplusplus
}
#endif

#endif /* KMEANS_INPUT_H */
