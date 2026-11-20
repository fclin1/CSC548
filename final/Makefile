CC ?= gcc
NVCC ?= nvcc
CUDA_ARCH ?= sm_61

CFLAGS := -O3 -std=c11 -D_POSIX_C_SOURCE=200809L -march=native -Wall -Wextra -Iinclude
LDFLAGS := -lm
NVCCFLAGS := -O3 -arch=$(CUDA_ARCH) --fmad=false -lineinfo -Iinclude -Xcompiler "-Wall -Wextra"

BIN_DIR := bin
OBJ_DIR := obj
SRC_DIR := src
INC_DIR := include

TARGETS := $(BIN_DIR)/csv_to_bin \
           $(BIN_DIR)/kmeans_seq \
           $(BIN_DIR)/kmeans_seq_prof \
           $(BIN_DIR)/kmeans_cuda_naive \
           $(BIN_DIR)/kmeans_cuda_shmem \
           $(BIN_DIR)/kmeans_cuda_device

.PHONY: all cpu prof cuda test clean dirs

all: dirs $(TARGETS)

cpu: dirs $(BIN_DIR)/csv_to_bin $(BIN_DIR)/kmeans_seq

prof: dirs $(BIN_DIR)/kmeans_seq_prof

cuda: dirs $(BIN_DIR)/kmeans_cuda_naive $(BIN_DIR)/kmeans_cuda_shmem $(BIN_DIR)/kmeans_cuda_device

dirs:
	@mkdir -p $(BIN_DIR) $(OBJ_DIR)

$(BIN_DIR)/csv_to_bin: $(SRC_DIR)/csv_to_bin.c $(INC_DIR)/config.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/kmeans_seq: $(SRC_DIR)/kmeans_seq.c $(INC_DIR)/*.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/kmeans_seq_prof: $(SRC_DIR)/kmeans_seq.c $(INC_DIR)/*.h
	$(CC) $(CFLAGS) -pg -fno-inline $< -o $@ $(LDFLAGS)

$(OBJ_DIR)/main_cuda.o: $(SRC_DIR)/main_cuda.c $(INC_DIR)/*.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/kmeans_cuda_%.o: $(SRC_DIR)/kmeans_cuda_%.cu $(INC_DIR)/kmeans_cuda.h $(INC_DIR)/config.h $(INC_DIR)/kmeans_types.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN_DIR)/kmeans_cuda_%: $(OBJ_DIR)/main_cuda.o $(OBJ_DIR)/kmeans_cuda_%.o
	$(NVCC) -arch=$(CUDA_ARCH) $^ -o $@ $(LDFLAGS)

test: all
	@bash tests/run_tests.sh

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR) gmon.out
