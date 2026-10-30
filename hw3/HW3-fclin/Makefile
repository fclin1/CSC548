# The harness is host C and is built with gcc; only your .cu goes through nvcc,
# and nvcc links (it pulls in the CUDA runtime).
CC     = gcc
NVCC  ?= nvcc

# ARC's GPU nodes are V100 (compute capability 7.0). Override for another card:
#   make spmv_cuda_scalar GPU_ARCH=sm_80
GPU_ARCH ?= sm_70

FLAG    = -O3 -std=gnu99 -I./include -Wno-unused-result -Wno-write-strings
NVFLAG  = -O3 -arch=$(GPU_ARCH) -I./include
LDFLAG  = -O3
LIBS    = -lm

SRC = src
KERNEL_DIR ?= $(SRC)

# All three implementations are required, one binary each. Overridable to build
# someone else's file, or use KERNEL_DIR for a flat submission directory:
#   make spmv_cuda_vector VECTOR_SRC=../solution/spmv-cuda-vector.cu
SCALAR_SRC ?= $(KERNEL_DIR)/spmv-cuda-scalar.cu
VECTOR_SRC ?= $(KERNEL_DIR)/spmv-cuda-vector.cu
ELL_SRC    ?= $(KERNEL_DIR)/spmv-cuda-ell.cu

HEADERS = $(wildcard include/*.h)

# Build the CPU baselines and all three required CUDA kernels.
all: spmv spmv_csr spmv_cuda_scalar spmv_cuda_vector spmv_cuda_ell

# ---- sequential baselines (plain gcc, no CUDA) ----
spmv: $(SRC)/spmv.o $(SRC)/mmio.o
	$(CC) $(LDFLAG) -o $@ $^ $(LIBS)

spmv_csr: $(SRC)/spmv_csr.o $(SRC)/mmio.o
	$(CC) $(LDFLAG) -o $@ $^ $(LIBS)

%.o: %.c $(HEADERS)
	$(CC) -o $@ -c $(FLAG) $<

# ---- CUDA versions: gcc harness objects, nvcc kernel object, nvcc link ----
HARNESS_OBJ = $(SRC)/main_cuda.o $(SRC)/mmio.o

spmv_cuda_scalar: $(HARNESS_OBJ) $(SCALAR_SRC:.cu=.cu.o)
	$(NVCC) $(NVFLAG) -o $@ $^ $(LIBS)

spmv_cuda_vector: $(HARNESS_OBJ) $(VECTOR_SRC:.cu=.cu.o)
	$(NVCC) $(NVFLAG) -o $@ $^ $(LIBS)

spmv_cuda_ell: $(HARNESS_OBJ) $(ELL_SRC:.cu=.cu.o)
	$(NVCC) $(NVFLAG) -o $@ $^ $(LIBS)

# The object name is derived from the source path, so pointing a target at a
# different file never links a stale object from the previous one.
%.cu.o: %.cu $(HEADERS)
	$(NVCC) -o $@ -c $(NVFLAG) $<

.PHONY: clean all
clean:
	find ./ -name "*.o" -delete
	rm -f spmv spmv_csr spmv_cuda_scalar spmv_cuda_vector spmv_cuda_ell
