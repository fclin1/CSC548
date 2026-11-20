#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "config.h"

#define MAX_LINE_LEN 65536
#define CHUNK_ROWS   8192

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --in <csv_file|-> --out <bin_file> [options]\n"
        "Options:\n"
        "  --delim <char>       Delimiter character (default: ',')\n"
        "  --skip-cols <int>    Number of leading columns to skip (default: 1)\n"
        "  --skip-header        Skip the first line (header line)\n"
        "  --skip-missing       Skip rows containing '?' or empty tokens\n"
        "  --max-rows <int>     Maximum rows to convert (default: 0 = all)\n"
        "  --help, -h           Print this help message\n",
        prog);
}

int main(int argc, char **argv) {
    const char *in_path = NULL;
    const char *out_path = NULL;
    char delim = ',';
    int skip_cols = 1;
    bool skip_header = false;
    bool skip_missing = false;
    int max_rows = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--in") == 0 && i + 1 < argc) {
            in_path = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--delim") == 0 && i + 1 < argc) {
            delim = argv[++i][0];
        } else if (strcmp(argv[i], "--skip-cols") == 0 && i + 1 < argc) {
            skip_cols = (int)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--skip-header") == 0) {
            skip_header = true;
        } else if (strcmp(argv[i], "--skip-missing") == 0) {
            skip_missing = true;
        } else if (strcmp(argv[i], "--max-rows") == 0 && i + 1 < argc) {
            max_rows = (int)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return EXIT_OK;
        } else {
            fprintf(stderr, "Unknown or incomplete argument '%s'\n", argv[i]);
            print_usage(argv[0]);
            return EXIT_CLI_ERR;
        }
    }

    if (!in_path || !out_path) {
        fprintf(stderr, "Error: Both --in and --out are required.\n");
        print_usage(argv[0]);
        return EXIT_CLI_ERR;
    }

    FILE *in_fp = NULL;
    if (strcmp(in_path, "-") == 0) {
        in_fp = stdin;
    } else {
        in_fp = fopen(in_path, "r");
        if (!in_fp) {
            fprintf(stderr, "Error: Cannot open input CSV file '%s'\n", in_path);
            return EXIT_IO_ERR;
        }
    }

    FILE *out_fp = fopen(out_path, "wb");
    if (!out_fp) {
        fprintf(stderr, "Error: Cannot open output binary file '%s'\n", out_path);
        if (in_fp != stdin) fclose(in_fp);
        return EXIT_IO_ERR;
    }

    /* Write 16-byte header placeholder: uint64_t N, uint64_t D */
    uint64_t header_placeholder[2] = { 0, 0 };
    if (fwrite(header_placeholder, sizeof(uint64_t), 2, out_fp) != 2) {
        fprintf(stderr, "Error: Failed to write binary header placeholder\n");
        if (in_fp != stdin) fclose(in_fp);
        fclose(out_fp);
        return EXIT_IO_ERR;
    }

    char line_buf[MAX_LINE_LEN];
    if (skip_header) {
        if (!fgets(line_buf, sizeof(line_buf), in_fp)) {
            fprintf(stderr, "Error: Could not read header line to skip\n");
            if (in_fp != stdin) fclose(in_fp);
            fclose(out_fp);
            return EXIT_IO_ERR;
        }
    }

    uint64_t num_points = 0;
    uint64_t num_dims = 0;
    float row_buffer[MAX_SUPPORTED_D];

    while (fgets(line_buf, sizeof(line_buf), in_fp)) {
        if (max_rows > 0 && num_points >= (uint64_t)max_rows) {
            break;
        }

        /* Check for missing token if requested */
        if (skip_missing) {
            if (strchr(line_buf, '?') != NULL) {
                continue;
            }
        }

        /* Tokenize line by delim */
        char *ptr = line_buf;
        int col_idx = 0;
        int dim_idx = 0;
        bool row_valid = true;

        while (*ptr != '\0') {
            char *start = ptr;
            while (*ptr != '\0' && *ptr != delim && *ptr != '\r' && *ptr != '\n') {
                ptr++;
            }
            char save_ch = *ptr;
            *ptr = '\0';

            if (col_idx >= skip_cols) {
                if (skip_missing && (ptr == start)) {
                    row_valid = false;
                    break;
                }
                char *endptr = NULL;
                float val = strtof(start, &endptr);
                if (endptr == start || !isfinite(val)) {
                    if (skip_missing) {
                        row_valid = false;
                        break;
                    } else {
                        fprintf(stderr, "Error: Invalid float '%s' at row %lu, col %d\n", start, (unsigned long)num_points, col_idx);
                        if (in_fp != stdin) fclose(in_fp);
                        fclose(out_fp);
                        return EXIT_DATA_ERR;
                    }
                }
                if (dim_idx >= MAX_SUPPORTED_D) {
                    fprintf(stderr, "Error: Dimensionality exceeds limit of %d\n", MAX_SUPPORTED_D);
                    if (in_fp != stdin) fclose(in_fp);
                    fclose(out_fp);
                    return EXIT_DATA_ERR;
                }
                row_buffer[dim_idx++] = val;
            }

            col_idx++;
            if (save_ch == '\0' || save_ch == '\r' || save_ch == '\n') {
                break;
            }
            ptr++;
        }

        if (!row_valid) {
            continue;
        }

        if (num_points == 0) {
            num_dims = (uint64_t)dim_idx;
            if (num_dims == 0) {
                fprintf(stderr, "Error: Zero feature dimensions parsed from first row\n");
                if (in_fp != stdin) fclose(in_fp);
                fclose(out_fp);
                return EXIT_DATA_ERR;
            }
        } else if ((uint64_t)dim_idx != num_dims) {
            fprintf(stderr, "Error: Dimension mismatch at row %lu (expected %lu, got %d)\n",
                    (unsigned long)num_points, (unsigned long)num_dims, dim_idx);
            if (in_fp != stdin) fclose(in_fp);
            fclose(out_fp);
            return EXIT_DATA_ERR;
        }

        if (fwrite(row_buffer, sizeof(float), (size_t)num_dims, out_fp) != (size_t)num_dims) {
            fprintf(stderr, "Error: Failed to write point data to '%s'\n", out_path);
            if (in_fp != stdin) fclose(in_fp);
            fclose(out_fp);
            return EXIT_IO_ERR;
        }

        num_points++;
    }

    if (in_fp != stdin) {
        fclose(in_fp);
    }

    /* Seek back and write final header */
    if (fseek(out_fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Error: Failed to rewind output file for header update\n");
        fclose(out_fp);
        return EXIT_IO_ERR;
    }

    uint64_t final_header[2] = { num_points, num_dims };
    if (fwrite(final_header, sizeof(uint64_t), 2, out_fp) != 2) {
        fprintf(stderr, "Error: Failed to write final header\n");
        fclose(out_fp);
        return EXIT_IO_ERR;
    }

    fclose(out_fp);

    printf("Successfully wrote %lu points with %lu dimensions to '%s' (%lu bytes)\n",
           (unsigned long)num_points, (unsigned long)num_dims, out_path,
           (unsigned long)(16 + num_points * num_dims * sizeof(float)));

    return EXIT_OK;
}
