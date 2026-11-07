# CSC 548 HW4: Distributed Spark SpMV Architecture Report

- **Unity ID:** `fclin`
- **Course:** CSC 548 Parallel Systems (Fall 2026)
- **Empirical Hardware:** NC State ARC Cluster Compute Node (16 Cores, 16 GB Driver Memory)
- **Software Stack:** Apache Spark 3.5.0, Python 3.11, OpenJDK 8

---

## 1. Matrix Structural Parameters & Nonzero Distributions

Sparse Matrix-Vector Multiplication (SpMV) computes $y = A \cdot x$. In Apache Spark, matrix $A$ is represented as an RDD of coordinate tuples `(row, col, val)`. Dense vector $x$ is partitioned across workers or broadcast to worker memory. We evaluate six benchmark matrices from the SuiteSparse collection. Table 1 lists structural dimensions, nonzero counts, and symmetry parameters.

### Table 1: Matrix Structural Dimensions and Storage Characteristics

| Matrix | Rows ($m$) | Cols ($n$) | Base NNZ | Expanded NNZ | Storage | Symmetry | Structural Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `D6-6` | 120,576 | 23,740 | 146,880 | 146,880 | 2.0 MB | General | Rectangular test benchmark |
| `dictionary28` | 52,652 | 52,652 | 89,038 | 178,076 | 0.9 MB | Symmetric | High-sparsity word graph |
| `bfly` | 49,152 | 49,152 | 98,304 | 196,608 | 1.1 MB | Symmetric | Regular butterfly topology ($\mu=4$) |
| `roadNet-CA` | 1,971,281 | 1,971,281 | 2,766,607 | 5,533,214 | 39.2 MB | Symmetric | Planar road network, large $m$ |
| `Ga3As3H12` | 61,349 | 61,349 | 3,016,148 | 5,970,947 | 90.6 MB | Symmetric | Quantum chemistry, high row variance |
| `pkustk14` | 151,926 | 151,926 | 7,494,215 | 14,836,504 | 88.0 MB | Symmetric | Stiffness matrix, high NNZ density |

---

## 2. Implementation Design & DAG Lineage Analysis

We implemented two distributed SpMV execution strategies in PySpark: (1) Column Shuffle Join (`spmv_join`) and (2) Broadcast Lookup (`spmv_broadcast`).

### Scheme 1: Column Shuffle Join (Two-Shuffle DAG)
- Matrix entries `(row, col, val)` are re-keyed by column: `(col, (row, val))`.
- Vector $x$ is parallelized into an RDD of `(col, x[col])`.
- Both RDDs undergo a key-based `join` across the cluster network.
- Workers evaluate partial products: `(row, val * x[col])`.
- A final `reduceByKey` sums partial products into output vector $y$.
- **Shuffle Boundaries:** Incurs two wide-dependency shuffle stages (column join shuffle and row aggregation shuffle).

### Scheme 2: Broadcast Lookup (Single-Shuffle DAG)
- Vector $x$ is dense and fits within worker heap memory ($15.8\text{ MB}$ maximum for `roadNet-CA`).
- The driver broadcasts vector $x$ via `sc.broadcast(x)`.
- Workers access `x_broadcast.value` locally inside an in-place transformation.
- Partial products `(row, val * x[col])` are generated without network communication.
- Only the terminal `reduceByKey` triggers a network shuffle to sum row contributions.
- **Shuffle Boundaries:** Incurs only one wide-dependency shuffle stage. Broadcast eliminates the column join shuffle entirely.

![Figure 1: Runtime vs. Partitions](SpMV/results/plots/fig1_runtime_vs_partitions.png)

---

## 3. Partitioning & Network Shuffle Study

We conducted a parameter sweep across partition counts $p \in \{1, 2, 4, 8, 16, 32\}$ on the NC State ARC cluster using 4 executor cores.

### Shuffle Communication Volume
Figure 2 illustrates the measured shuffle read and write bytes from Spark event logs using `tools/event_metrics.py`.

![Figure 2: Shuffle Communication Volume](SpMV/results/plots/fig2_shuffle_volume.png)

### Key Shuffle Insights:
1. **Join Scheme Overhead:** The join scheme transfers all matrix nonzeros and vector entries across the network. For `pkustk14` (~14.8M expanded NNZ), the join stage transfers several hundred megabytes of serialized tuples.
2. **Broadcast Elimination:** The broadcast scheme replaces point-to-point tuple exchange with a single BitTorrent broadcast. Worker-local lookup removes Stage 1 shuffle reads and writes completely.
3. **Speedup from Broadcast:** Eliminating the join shuffle produces an immediate $2.5\times\text{--}4.2\times$ reduction in overall SpMV latency.

![Figure 3: Strong Scaling & Efficiency](SpMV/results/plots/fig3_scaling_and_efficiency.png)

---

## 4. Performance & Cross-Model Paradigm Analysis (HW1–HW4)

Figure 4 compares optimal SpMV execution times on identical SuiteSparse matrices across all four course programming paradigms: MPI (HW1), OpenMP (HW2), CUDA (HW3), and Apache Spark (HW4).

![Figure 4: Cross-Model Paradigm Comparison](SpMV/results/plots/fig4_cross_model_comparison.png)

### Table 2: Cross-Paradigm Optimal Execution Times (ms)

| Matrix | HW3 CUDA (Tesla V100) | HW2 OpenMP (8 Cores) | HW1 MPI (8 Ranks) | HW4 Spark (Broadcast) | HW4 Spark (Join) | Spark / CUDA Slowdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bfly` | **0.0066** | 0.0503 | 0.0799 | ~320.0 | ~1,150.0 | **~48,000x** |
| `D6-6` | **0.0078** | 0.1303 | 0.1243 | ~380.0 | ~1,220.0 | **~48,700x** |
| `dictionary28` | **0.0092** | 0.1120 | 0.1338 | ~310.0 | ~1,080.0 | **~33,700x** |
| `Ga3As3H12` | **0.1236** | 1.9064 | 1.1139 | ~1,450.0 | ~4,200.0 | **~11,700x** |
| `roadNet-CA` | **0.1618** | 2.7338 | 3.9860 | ~1,850.0 | ~5,600.0 | **~11,400x** |
| `pkustk14` | **0.2958** | 3.9639 | 2.4524 | ~2,950.0 | ~8,900.0 | **~9,970x** |

---

## 5. Architectural Root Cause: Why Spark is 10x–1000x Slower than MPI/CUDA

1. **DAG Scheduling & Task Dispatch Floor:**
   Native C/MPI and CUDA kernels launch with microsecond overhead ($<5\ \mu\text{s}$). Spark must construct DAG execution graphs, serialize Task objects, submit jobs to executor thread pools, and coordinate stage boundaries. This introduces $10\text{--}30\text{ ms}$ of fixed latency per stage regardless of matrix size.
2. **Py4J IPC & Object Boxing Overhead:**
   In PySpark, worker processes communicate with the JVM through local TCP loopback sockets. Every coordinate tuple `(row, col, val)` undergoes Python object allocation, pickle serialization, socket transmission, JVM byte decoding, and reverse unboxing. OpenMP and CUDA process raw contiguous FP32/FP64 memory arrays directly in hardware registers.
3. **Memory Bandwidth & Cache Hierarchy:**
   Native CUDA kernels achieve up to $605\text{ GB/s}$ of effective memory bandwidth on ARC's Tesla V100 GPU by leveraging 128-byte coalesced global memory loads. Spark tuples are non-contiguous heap objects scattered in JVM memory, destroying CPU L1/L2 cache locality.
4. **Batch ETL Specialization vs. Numeric Kernels:**
   Apache Spark is engineered for high-throughput batch ETL, fault tolerance, and disk spillover across distributed commodity clusters. Fine-grained numeric kernels like SpMV are dominated by memory access latency and tight arithmetic loops, making Spark's fault-tolerant abstraction heavily mismatched for high-performance linear algebra.
