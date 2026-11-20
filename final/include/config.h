#ifndef KMEANS_CONFIG_H
#define KMEANS_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/* Standardized Exit Codes */
#define EXIT_OK          0
#define EXIT_CLI_ERR     1
#define EXIT_IO_ERR      2
#define EXIT_DATA_ERR    3
#define EXIT_CUDA_ERR    4
#define EXIT_VERIFY_ERR  5

/* Hardware and Kernel Constants */
#define SHMEM_MAX_FLOATS        12288   /* 48 KB / 4 bytes (Pascal GP102 SMEM per block) */
#define ACCUM_SHMEM_MAX_FLOATS  6144    /* 24 KB / 4 bytes (Path A limit for block-private reduction) */
#define DEFAULT_BLOCK_SIZE      256
#define MAX_BLOCK_SIZE          1024
#define MAX_SUPPORTED_D         12288

/* Verification and Parity Tolerances */
#define VERIFY_REL_L2_TOL       1e-4
#define VERIFY_LABEL_MATCH_MIN  99.99

/* Math Constant Fallback for strict C11 */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
}
#endif

#endif /* KMEANS_CONFIG_H */
