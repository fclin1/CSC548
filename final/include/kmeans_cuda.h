#ifndef KMEANS_CUDA_H
#define KMEANS_CUDA_H

#include "kmeans_types.h"

#ifdef __cplusplus
extern "C" {
#endif

const char *kmeans_cuda_impl_name(void);

void kmeans_cuda_run(
    const float *h_points_aos,
    const float *h_init_centroids,
    float *h_final_centroids,
    int *h_assignments,
    const KMeansConfig *cfg,
    KMeansMetrics *metrics);

#ifdef __cplusplus
}
#endif

#endif /* KMEANS_CUDA_H */
