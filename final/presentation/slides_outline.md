# Parallel K-Means Clustering on NVIDIA GPUs: Pure C11 & CUDA

**Course:** CSC 548 (Parallel Systems, Fall 2026)  
**Author:** Frank Lin  
**Target Hardware:** Intel Core i5-9600K CPU + NVIDIA GeForce GTX 1080 Ti GPU (Pascal `sm_61`)  
**Presentation Time Limit:** 4 Minutes (+ 30 Seconds Q&A)

---

## Slide 1: Problem Statement, Workload & Motivation (0:00 – 1:00)

### Visual Content
* Diagram of Lloyd's K-Means clustering algorithm: iterative Expectation (Assignment) and Maximization (Centroid Update).
* Computational complexity: $\mathcal{O}(I \cdot N \cdot K \cdot D)$ FLOPs.
* Real-world dataset: UCI SUSY ($N = 5{,}000{,}000$ points, $D = 18$ features, $360\text{ MB}$ single-precision binary footprint).

### Speaker Script (1 Minute)
> "Good morning everyone. Today I'm presenting our parallel acceleration of Lloyd's K-Means clustering on the GPU. K-Means is a cornerstone of unsupervised learning, widely applied in recommendation systems, anomaly detection, and scientific data analysis.
>
> However, on massive datasets like the UCI SUSY benchmark—with 5 million points and 18 continuous dimensions—a single iteration requires evaluating 5 million Euclidean distances against all $K$ centroids. On a serial CPU, this workload takes hours for large cluster counts.
>
> Our goal was simple: achieve strong scaling and maximum throughput on a commodity NVIDIA GTX 1080 Ti GPU while maintaining strict deterministic numerical correctness against a compiler-optimized serial baseline."

---

## Slide 2: Serial Baseline & Profiling-Driven Insights (1:00 – 2:00)

### Visual Content
* Table of hardware roofline specifications: Intel i5-9600K (128 GFLOP/s, 41.6 GB/s) vs. NVIDIA GTX 1080 Ti (11,340 GFLOP/s, 484 GB/s).
* Pie chart and table from `gprof` profiling output:
  * `kmeans_cpu_assign`: **97.13%** of total runtime.
  * `kmeans_cpu_update`: **2.15%** of runtime.
  * `validate_dataset` and overhead: **0.72%**.

### Speaker Script (1 Minute)
> "In accordance with core parallel systems principles, we built a fair, compiler-optimized serial C11 baseline first, compiled with `-O3 -march=native`.
>
> Following the rule of 'never optimize blind,' we profiled the serial execution using `gprof` and hardware performance timers on 1 million SUSY points. The profiling data confirmed a massive hot spot: **97.13%** of execution time was spent solely inside the distance assignment loop.
>
> Because distance calculations for each data point are completely independent, this step exhibits embarrassingly data-parallel characteristics. But according to Amdahl's Law, even if we speed up assignment by $100\times$, the remaining 3% in centroid updates and data movement would cap total speedup at roughly $25\times$ unless we parallelize the entire pipeline on-device."

---

## Slide 3: Three-Stage Progressive CUDA Optimization (2:00 – 3:00)

### Visual Content
* Architectural diagram contrasting the 3 CUDA stages:
  1. **Stage 1 (Naive):** 1 thread/point, AoS row-major layout, global DRAM centroids, per-iteration D2H/H2D PCIe transfers.
  2. **Stage 2 (Shared Memory + Coalescing):** 128-float padded SoA layout for 100% coalesced 128-byte DRAM reads; tiled shared-memory centroid broadcast (0 bank conflicts).
  3. **Stage 3 (Full On-Device):** Vectorized `float4` loads (`LDG.E.128`) with 4-way register reuse; two-tier block-shared atomic reduction; `__shfl_down_sync` warp-shuffle convergence; zero per-iteration PCIe transfers.

### Speaker Script (1 Minute)
> "To understand where parallel performance comes from, we designed a progressive three-stage CUDA architecture.
>
> In Stage 1, we implemented a naive 1-thread-per-point kernel using the standard Row-Major Array-of-Structures layout. While this offloaded computation, adjacent threads suffered from uncoalesced memory reads, and per-iteration PCIe transfers created a severe Amdahl bottleneck.
>
> In Stage 2, we addressed memory coalescing by transforming data into a 128-float padded Structure-of-Arrays (SoA). This ensured every warp accessed consecutive 128-byte cache lines. Furthermore, we staged centroids into fast on-chip shared memory, broadcasting coordinates across all 32 warp lanes with zero bank conflicts. This yielded an immediate $6.4\times$ speedup on the assignment kernel alone.
>
> Finally, in Stage 3, we eliminated host-device synchronization entirely. We vectorized loads using `float4` instructions with 4-way register reuse, and performed centroid recomputation directly on the GPU using hierarchical block-private reductions and Pascal-safe warp shuffles."

---

## Slide 4: Empirical Results, Scaling & Conclusions (3:00 – 4:00)

### Visual Content
* Speedup curve vs. $N$ (showing up to **38.99x speedup** on 1M points and **33.1x** on 5M points).
* Execution time breakdown chart: showing Stage 3 completely eliminating the PCIe transfer bar.
* Empirical Roofline chart: GTX 1080 Ti ceiling vs. attained GFLOP/s across $K \in [8, 1024]$, marking the ridge point at $K \approx 33$ and 48 KB shared-memory cliff at $K = 682$.

### Speaker Script (1 Minute)
> "Here are our measured empirical results on the GTX 1080 Ti.
>
> On 1 million SUSY points, the serial CPU baseline took 2,724 ms per 10 iterations. Stage 1 dropped this to 400 ms. Stage 2 reached 133 ms. Stage 3 achieved **69.86 ms**—an overall speedup of **38.99x**, achieving **249.9 GFLOP/s** and **10.88 GB/s** of effective bandwidth.
>
> On the Roofline model, sweeping $K$ from 8 to 1024 demonstrates a clear regime shift: for $K < 33$, execution is memory-bandwidth bound. Beyond $K \approx 33$, arithmetic intensity exceeds the GTX 1080 Ti ridge point ($23.4\text{ FLOP/byte}$), allowing GPU throughput to scale into hundreds of GFLOP/s.
>
> Crucially, our automated verification test suite confirmed that all parallel implementations match the serial ground truth within $10^{-6}$ relative $L_2$ error and 100% label match.
>
> Thank you, and I am happy to take your questions."

---

## Slide 5: Anticipated 30-Second Q&A Defense

**Q1: How do you handle empty clusters if no points are assigned to a centroid?**  
> *Answer:* "We implement a deterministic empty-cluster retention rule across both CPU and GPU: if a cluster's point count is zero, the centroid retains its previous coordinates and reports zero shift, preventing division by zero or NaN propagation."

**Q2: Did you observe any numerical divergence between CPU and GPU results?**  
> *Answer:* "Initially, hardware fused multiply-add (FMA) contractions produced minor rounding differences after 20 iterations. By adding `--fmad=false` during compilation, we achieved exact bit-level floating-point parity with 100.0000% label match against the serial baseline."

**Q3: How does your implementation scale if cluster count $K$ exceeds shared memory capacity?**  
> *Answer:* "Our kernel automatically tiles centroids in chunks of $12{,}288 / D$ floats ($682$ centroids for SUSY). For $K=1024$, it executes a two-pass tile loop seamlessly while maintaining zero bank conflicts."
