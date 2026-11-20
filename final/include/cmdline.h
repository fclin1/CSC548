#ifndef KMEANS_CMDLINE_H
#define KMEANS_CMDLINE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include "config.h"
#include "kmeans_types.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline void init_default_config(KMeansConfig *cfg) {
    memset(cfg, 0, sizeof(KMeansConfig));
    cfg->file_path[0] = '\0';
    cfg->synthetic = false;
    cfg->n = 0;
    cfg->d = 18;
    cfg->k = 32;
    cfg->max_iters = 20;
    cfg->fixed_iters = true;
    cfg->early_stop = false;
    cfg->tol = 1e-5f;
    cfg->seed = 42;
    cfg->block_size = DEFAULT_BLOCK_SIZE;
    cfg->warmup = 1;
    cfg->trials = 1;
    cfg->verify = false;
    cfg->dump_ref_prefix[0] = '\0';
    cfg->check_ref_prefix[0] = '\0';
    cfg->csv_path[0] = '\0';
    cfg->csv_header = false;
}

static inline void print_usage(const char *prog_name) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  --file <path>         Path to binary dataset file (.bin)\n"
        "  --synthetic           Generate synthetic Gaussian clusters in memory\n"
        "  --n <int>             Number of points (0 = all if --file given)\n"
        "  --d <int>             Dimensionality (default: 18)\n"
        "  --k <int>             Number of clusters (default: 32)\n"
        "  --iters <int>         Maximum iterations (default: 20)\n"
        "  --fixed-iters         Run fixed iteration count (default: enabled)\n"
        "  --early-stop          Stop early if centroid shift <= tol^2\n"
        "  --tol <float>         Early-stop tolerance (default: 1e-5)\n"
        "  --seed <uint>         Seed for PRNG and initial centroids (default: 42)\n"
        "  --block-size <int>    CUDA block size (multiple of 32, default: 256)\n"
        "  --warmup <int>        Untimed warmup trials (default: 1)\n"
        "  --trials <int>        Timed benchmark trials (default: 1)\n"
        "  --verify              Run in-process CPU reference verification\n"
        "  --dump-ref <prefix>   Dump reference centroids and labels to disk\n"
        "  --check-ref <prefix>  Validate against dumped reference files\n"
        "  --csv <path>          Append 21-column metrics row to CSV file\n"
        "  --csv-header          Print CSV header line and exit 0\n"
        "  --help, -h            Print this help message and exit 0\n",
        prog_name);
}

static inline int parse_args(int argc, char **argv, KMeansConfig *cfg) {
    init_default_config(cfg);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(EXIT_OK);
        } else if (strcmp(argv[i], "--csv-header") == 0) {
            cfg->csv_header = true;
            return EXIT_OK;
        } else if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
            strncpy(cfg->file_path, argv[++i], sizeof(cfg->file_path) - 1);
            cfg->file_path[sizeof(cfg->file_path) - 1] = '\0';
        } else if (strcmp(argv[i], "--synthetic") == 0) {
            cfg->synthetic = true;
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            cfg->n = (int)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--d") == 0 && i + 1 < argc) {
            cfg->d = (int)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            cfg->k = (int)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            cfg->max_iters = (int)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--fixed-iters") == 0) {
            cfg->fixed_iters = true;
            cfg->early_stop = false;
        } else if (strcmp(argv[i], "--early-stop") == 0) {
            cfg->early_stop = true;
            cfg->fixed_iters = false;
        } else if (strcmp(argv[i], "--tol") == 0 && i + 1 < argc) {
            cfg->tol = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg->seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            cfg->block_size = (int)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            cfg->warmup = (int)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) {
            cfg->trials = (int)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--verify") == 0) {
            cfg->verify = true;
        } else if (strcmp(argv[i], "--dump-ref") == 0 && i + 1 < argc) {
            strncpy(cfg->dump_ref_prefix, argv[++i], sizeof(cfg->dump_ref_prefix) - 1);
            cfg->dump_ref_prefix[sizeof(cfg->dump_ref_prefix) - 1] = '\0';
        } else if (strcmp(argv[i], "--check-ref") == 0 && i + 1 < argc) {
            strncpy(cfg->check_ref_prefix, argv[++i], sizeof(cfg->check_ref_prefix) - 1);
            cfg->check_ref_prefix[sizeof(cfg->check_ref_prefix) - 1] = '\0';
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            strncpy(cfg->csv_path, argv[++i], sizeof(cfg->csv_path) - 1);
            cfg->csv_path[sizeof(cfg->csv_path) - 1] = '\0';
        } else {
            fprintf(stderr, "Error: Unknown or incomplete CLI argument '%s'\n", argv[i]);
            print_usage(argv[0]);
            return EXIT_CLI_ERR;
        }
    }

    /* Basic CLI validation */
    if (!cfg->synthetic && cfg->file_path[0] == '\0') {
        fprintf(stderr, "Error: Either --file <path> or --synthetic must be specified.\n");
        return EXIT_CLI_ERR;
    }
    if (cfg->k <= 0) {
        fprintf(stderr, "Error: --k must be strictly positive (got %d).\n", cfg->k);
        return EXIT_CLI_ERR;
    }
    if (cfg->max_iters <= 0) {
        fprintf(stderr, "Error: --iters must be strictly positive (got %d).\n", cfg->max_iters);
        return EXIT_CLI_ERR;
    }
    if (cfg->block_size <= 0 || cfg->block_size > MAX_BLOCK_SIZE || (cfg->block_size % 32 != 0)) {
        fprintf(stderr, "Error: --block-size must be a multiple of 32 between 32 and %d (got %d).\n",
                MAX_BLOCK_SIZE, cfg->block_size);
        return EXIT_CLI_ERR;
    }
    if (cfg->synthetic) {
        if (cfg->n <= 0) {
            fprintf(stderr, "Error: --synthetic requires --n > 0 (got %d).\n", cfg->n);
            return EXIT_CLI_ERR;
        }
        if (cfg->d <= 0 || cfg->d > MAX_SUPPORTED_D) {
            fprintf(stderr, "Error: Dimensionality --d must be in [1, %d] (got %d).\n",
                    MAX_SUPPORTED_D, cfg->d);
            return EXIT_DATA_ERR;
        }
        if (cfg->n < cfg->k) {
            fprintf(stderr, "Error: Point count N (%d) must be >= cluster count K (%d).\n",
                    cfg->n, cfg->k);
            return EXIT_DATA_ERR;
        }
    }

    return EXIT_OK;
}

#ifdef __cplusplus
}
#endif

#endif /* KMEANS_CMDLINE_H */
