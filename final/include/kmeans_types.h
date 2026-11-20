#ifndef KMEANS_TYPES_H
#define KMEANS_TYPES_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char file_path[512];
    bool synthetic;
    int n;
    int d;
    int k;
    int max_iters;
    bool fixed_iters;
    bool early_stop;
    float tol;
    uint32_t seed;
    int block_size;
    int warmup;
    int trials;
    bool verify;
    char dump_ref_prefix[512];
    char check_ref_prefix[512];
    char csv_path[512];
    bool csv_header;
} KMeansConfig;

typedef struct {
    int iters_run;
    double io_ms;
    double setup_ms;
    double assign_ms;
    double update_ms;
    double xfer_ms;
    double total_compute_ms;
    double ms_per_iter;
    double gflops;
    double gbytes_s;
} KMeansMetrics;

typedef struct {
    double centroid_rel_l2;
    double centroid_max_abs;
    double label_match_pct;
    bool passed;
} VerifyResult;

#ifdef __cplusplus
}
#endif

#endif /* KMEANS_TYPES_H */
