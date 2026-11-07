# NC State ARC Cluster Execution Guide: CSC 548 HW4 (Spark SpMV)

This guide provides instructions to run the **Spark SpMV benchmark suite** on the **NC State ARC Cluster** (Spark 3.5.0, Python 3.11, OpenJDK 8), pull the results back, generate all benchmark plots, compile the technical report PDF, and package the final submission archive `HW4-fclin.zip`.

---

## 1. Quick-Start Workflow

### Step 1: Connect to NC State VPN & Push Code to ARC
Ensure Cisco AnyConnect / Secure Client is connected to `vpn.ncsu.edu`. Then from your local repository root:
```bash
# Push HW4 code and scripts to ARC (matrices are automatically symlinked from HW3/HW2 on ARC)
bash hw4/sync_arc.sh push fclin

# (Optional: if matrices are missing on ARC, pass --with-matrices to upload them)
bash hw4/sync_arc.sh push fclin --with-matrices
```

---

### Step 2: Log into ARC and Verify Environment
```bash
ssh fclin@login.arc.ncsu.edu
cd ~/CSC548/hw4/SpMV

# Run the automated verification script (checks Python, Java, matrices, and runs smoke tests):
bash setup_arc.sh
```

---

### Step 3: Launch Benchmarks on ARC

Submit the Slurm jobs:
```bash
# 1. Main Sweep (72 configurations: 6 matrices x 2 schemes x 6 partition counts):
sbatch sweep.sbatch

# 2. Strong Scaling Core Sweep (cores 1, 2, 4, 8, 16 on representative matrices):
sbatch sweep_cores.sbatch
```

To monitor job status:
```bash
squeue -u fclin
```
Or from your local machine:
```bash
bash hw4/sync_arc.sh status fclin
```

---

### Step 4: Pull Benchmark Results to Local Machine
Once the Slurm jobs complete on ARC:
```bash
# Run on your local machine:
bash hw4/sync_arc.sh pull fclin
```
This downloads `spark.csv`, `shuffle.csv`, JSON logs, and event logs into `hw4/SpMV/results/`.

---

### Step 5: Generate Plots, Build Report & Package Archive
From the repository root on your local machine:
```bash
# 1. Generate Figures 1-4 (stored in hw4/SpMV/results/plots/):
python hw4/SpMV/generate_plots.py --results-dir hw4/SpMV/results --out-dir hw4/SpMV/results/plots

# 2. Build publication-quality report.pdf and HW4-fclin.zip:
python hw4/SpMV/build_report.py
```

The resulting submission zip will be generated at `hw4/HW4-fclin.zip`.
