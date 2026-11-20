#ifndef KMEANS_ALLOC_H
#define KMEANS_ALLOC_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdlib.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocate 64-byte aligned memory buffer for AVX2 cache-line alignment.
 */
static inline void *aligned_alloc_64(size_t bytes) {
    if (bytes == 0) {
        bytes = 64;
    }
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, bytes) != 0) {
        return NULL;
    }
    return ptr;
}

/**
 * Free memory allocated via aligned_alloc_64.
 */
static inline void aligned_free_64(void *ptr) {
    free(ptr);
}

#ifdef __cplusplus
}
#endif

#endif /* KMEANS_ALLOC_H */
