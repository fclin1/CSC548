#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

int main(void) {
    /* 1. Empty cluster fixture: 100 points, 4 dimensions */
    FILE *fp1 = fopen("data/test_empty_cluster.bin", "wb");
    if (!fp1) {
        perror("data/test_empty_cluster.bin");
        return 1;
    }
    uint64_t n1 = 100, d1 = 4;
    fwrite(&n1, sizeof(uint64_t), 1, fp1);
    fwrite(&d1, sizeof(uint64_t), 1, fp1);
    for (int i = 0; i < 50; ++i) {
        float pt[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        fwrite(pt, sizeof(float), 4, fp1);
    }
    for (int i = 0; i < 50; ++i) {
        float pt[4] = { 10.0f, 10.0f, 10.0f, 10.0f };
        fwrite(pt, sizeof(float), 4, fp1);
    }
    fclose(fp1);

    /* 2. Ties fixture: 100 identical points at origin */
    FILE *fp2 = fopen("data/test_ties.bin", "wb");
    if (!fp2) {
        perror("data/test_ties.bin");
        return 1;
    }
    uint64_t n2 = 100, d2 = 2;
    fwrite(&n2, sizeof(uint64_t), 1, fp2);
    fwrite(&d2, sizeof(uint64_t), 1, fp2);
    for (int i = 0; i < 100; ++i) {
        float pt[2] = { 0.0f, 0.0f };
        fwrite(pt, sizeof(float), 2, fp2);
    }
    fclose(fp2);

    /* 3. NaN fixture: 10 points with NaN injected */
    FILE *fp3 = fopen("data/test_nan.bin", "wb");
    if (!fp3) {
        perror("data/test_nan.bin");
        return 1;
    }
    uint64_t n3 = 10, d3 = 2;
    fwrite(&n3, sizeof(uint64_t), 1, fp3);
    fwrite(&d3, sizeof(uint64_t), 1, fp3);
    for (int i = 0; i < 9; ++i) {
        float pt[2] = { 1.0f, 2.0f };
        fwrite(pt, sizeof(float), 2, fp3);
    }
    float pt_nan[2] = { 1.0f, (float)NAN };
    fwrite(pt_nan, sizeof(float), 2, fp3);
    fclose(fp3);

    return 0;
}
