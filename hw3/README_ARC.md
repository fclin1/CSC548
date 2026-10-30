# NC State ARC GPU Cluster Guide: CSC 548 HW3 (CUDA SpMV)

This guide provides instructions to run the **CUDA SpMV benchmark suite** on an **NC State ARC GPU node** (NVIDIA Tesla V100, compute capability `sm_70`), generate all benchmark plots, compile the technical report PDF, and build the submission zip.

---

## 1. ARC Architecture Overview

- **Login Nodes:** `login.arc.ncsu.edu` or `login-gpu.arc.ncsu.edu`
- **GPU Partition:** `gpu`
- **GPU Accelerator:** NVIDIA Tesla V100 SXM2 / PCIe (Volta architecture, `sm_70`)
  - **CUDA Cores:** 5,120 FP32 cores
  - **Memory:** 16 GB / 32 GB HBM2
  - **Theoretical Peak Bandwidth:** 900 GB/s
  - **Peak Single-Precision Compute:** 14.1 TFLOP/s
  - **Roofline Knee:** $14,100 / 900 = 15.7\text{ FLOP/byte}$
- **Host CPU:** Dual-socket Intel Xeon Gold / Silver (supports AVX-512)

---

## 2. Quick-Start Workflow

### Step 1: Connect to ARC
```bash
ssh <unityid>@login.arc.ncsu.edu
```

### Step 2: Navigate to HW3 Workspace
Ensure your repository is located in your home directory or scratch space:
```bash
cd ~/CSC548/hw3/SpMV
```

If you need to push code and matrices from your local machine, run the sync helper on your **local machine**:
```bash
# On your local machine (from CSC548 repo root):
bash hw3/sync_arc.sh push <unityid> --with-matrices
```

### Step 3: Run Environment Setup
```bash
# On ARC (inside hw3/SpMV):
bash setup_arc.sh
```
This automated script:
1. Detects and loads `cuda`, `gcc`, and `python` modules.
2. Checks that `matrices/` contains the 6 benchmark matrices (`D6-6.mtx`, `dictionary28.mtx`, `bfly.mtx`, `Ga3As3H12.mtx`, `pkustk14.mtx`, `roadNet-CA.mtx`). If missing, it symlinks from `hw2/SpMV/matrices`.
3. Creates a Python virtual environment (`venv_arc`) and installs `matplotlib`, `pandas`, `reportlab`, and `numpy`.
4. Builds the kernels with `GPU_ARCH=sm_70`.
5. Runs a quick sanity smoke test on `bfly.mtx`.

---

## 3. Running the Benchmark Experiment

You can execute the benchmarks using either a **SLURM batch job** or an **interactive GPU session**.

### Option A: SLURM Batch Job (Recommended)

Submit the provided batch job script:
```bash
sbatch run_benchmarks.batch
```

**What the batch job executes:**
1. Loads CUDA and compiler modules.
2. Records CPU and GPU hardware topology into `results/hardware_info.txt`.
3. Compiles all binaries targeting `sm_70` (`spmv`, `spmv_csr`, `spmv_cuda_scalar`, `spmv_cuda_vector`, `spmv_cuda_ell`).
4. Runs single-core sequential CPU baseline across all 6 matrices (`results/cpu_baselines.txt`).
5. Sweeps all 3 kernels across 5 block sizes ($B \in \{64, 128, 256, 512, 1024\}$) and all 6 matrices (90 runs total), saving results to `results/all_benchmarks_combined.csv`.
6. Generates 5 high-resolution (300 DPI) figures in `results/plots/`.
7. Compiles the ASD-STE100 verified technical report into `report.pdf`.
8. Copies deliverables into `../HW3-fclin/` and creates `../HW3-fclin.zip`.

#### Monitoring Job Status:
```bash
# Check queue position / status:
squeue -u <unityid>

# View live output:
tail -f hw3_benchmarks.<job_id>.out

# Check for errors:
cat hw3_benchmarks.<job_id>.err
```

---

### Option B: Interactive GPU Session

If you prefer to run interactively, allocate a GPU node:
```bash
srun -p gpu --gres=gpu:1 -c 4 -t 01:00:00 --pty bash -i
```

Once your shell is active on the GPU node (e.g. `c0123`):
```bash
cd ~/CSC548/hw3/SpMV

# Run the complete automated interactive suite:
bash run_interactive.sh
```

Or run steps manually:
```bash
# 1. Load modules:
module load cuda gcc python

# 2. Compile for V100:
make clean && make all GPU_ARCH=sm_70

# 3. Run benchmarks:
bash run_experiments.sh

# 4. Generate plots and report:
venv_arc/bin/python3 generate_plots.py
venv_arc/bin/python3 build_report.py
```

---

## 4. Retrieving Deliverables to Your Local Machine

After the job completes on ARC, pull the updated deliverables and results back to your local repository.

From your **local machine terminal** (inside the `CSC548/` directory):
```bash
bash hw3/sync_arc.sh pull <unityid>
```

This updates:
- `hw3/HW3-fclin.zip` (Submission archive)
- `hw3/report.pdf` (Final technical report)
- `hw3/SpMV/results/all_benchmarks_combined.csv` (ARC V100 raw measurements)
- `hw3/SpMV/results/plots/*.png` (Figures plotted with live V100 data)

---

## 5. Verification Checklist

Before final submission on Moodle, verify:
1. [x] **Binaries compile cleanly:** `make clean && make all GPU_ARCH=sm_70` returns exit code 0.
2. [x] **Correctness tests pass:** All 6 matrices pass verification with relative $L_2$ error $\le 10^{-6}$.
3. [x] **Directory structure:** `HW3-fclin/` contains:
   - `spmv-cuda-scalar.cu`
   - `spmv-cuda-vector.cu`
   - `spmv-cuda-ell.cu`
   - `Makefile`
   - `report.pdf`
4. [x] **Archive created:** `HW3-fclin.zip` contains exactly the `HW3-fclin/` directory.
5. [x] **ASD-STE100 Compliance:** Run `python verify_ste100.py` to confirm 0 violations.

---

## 6. Troubleshooting & FAQs

### Q: `nvcc: command not found`
Run `module load cuda` (or `module load cuda/11.8.0` / `module load cuda/12.2`). Verify with `which nvcc`.

### Q: `Error: Matrix matrices/D6-6.mtx not found!`
Ensure you uploaded the matrices from HW2. Run `bash hw3/sync_arc.sh push <unityid> --with-matrices` from your local machine, or symlink them if HW2 exists on ARC:
```bash
ln -sfn ~/CSC548/hw2/SpMV/matrices ~/CSC548/hw3/SpMV/matrices
```

### Q: Does ELL layout run out of memory on `Ga3As3H12`?
No. The Tesla V100 has 16 GB (or 32 GB) HBM2 memory. `Ga3As3H12` requires 796 MB in ELL format, which easily fits within the V100's memory headroom. Memory allocation is protected with host and device allocation checks.

### Q: How long does the full benchmark suite take?
Single-core CPU baseline takes ~30 seconds total. The 90 GPU runs (6 matrices $\times$ 5 block sizes $\times$ 3 kernels) take approximately 1.5 to 2 minutes total on the V100.

