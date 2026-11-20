#ifndef KMEANS_TIMER_H
#define KMEANS_TIMER_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Return current wall-clock time in seconds using CLOCK_MONOTONIC.
 */
static inline double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#ifdef __cplusplus
}
#endif

#endif /* KMEANS_TIMER_H */
