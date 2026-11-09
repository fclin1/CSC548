# CSC 548: Parallel Systems — Final Project Guidelines Outline

**Instructor:** Jiajia Li  
**Course:** CSC 548 · Parallel Systems (Fall 2026)  
**Submission Portal:** Moodle  

---

## 1. Project Overview & Core Requirements

* **Core Objective:** Take a real computational workload, parallelize it, optimize it, and systematically evaluate and analyze performance.
* **Guiding Principle:** Open-ended application and parallel model, but strict adherence to engineering discipline:
  $$\text{Correct Baseline} \longrightarrow \text{Parallel Implementation} \longrightarrow \text{Rigorous Empirical Measurement}$$
* **Core Philosophy:** *"A small problem done thoroughly beats an ambitious one with no results."*

---

## 2. Project Tracks & Topic Categories

### Primary Tracks
1. **Accelerate (Recommended / Best Bet):** Speed up a real-world computational problem using parallel programming.
2. **Reproduce (Recommended / Best Bet):** Re-implement and evaluate a parallel algorithm or high-performance computing result from a published paper.
3. **Survey (Not Recommended):** Literature survey on a parallelism or performance topic.

### Alternative Acceptable Focus Areas
* **Performance Modeling:** Developing analytical performance models for a parallel algorithm.
* **Profiling & Bottleneck Analysis:** In-depth profiling study (must include at least one concrete optimization driven by the findings).
* **Cross-Platform Benchmarking:** Empirical performance comparison across multiple hardware architectures, scenarios, or configurations.
* **I/O Optimization:** Parallel I/O and data movement optimizations.

*(Note: All projects must produce a concrete artifact — an optimization, quantitative comparison, or empirical model — not merely qualitative observations.)*

### Instructor Research Candidate Topics (Block-Sparse Tensor Contraction / TIDES)
* **Multicore CPU:** OpenMP threading, SIMD vectorization (AVX/tuned micro-kernels/BLAS), static vs. dynamic scheduling for load imbalance.
* **Distributed Nodes:** MPI implementation, 1D block partitioning, sparse communication exchange vs. collective broadcasts, strong/weak scaling.
* **AMD GPU:** HIP / ROCm implementation, batched rocBLAS or custom tile kernels, profiling via rocprof/Omniperf.
* **Intel GPU:** SYCL / oneAPI (DPC++), oneMKL batched GEMM, profiling with VTune / Advisor.
* **Tenstorrent AI Accelerator:** TT-Metalium / TT-NN dataflow programming, tile mapping ($32 \times 32$), NoC utilization.

---

## 3. Implementation Constraints & Technical Rules

* **Programming Language:** **Strictly C or C++**.
* **Parallel Programming Models (Unrestricted):**
  * **Shared-Memory:** OpenMP, POSIX Threads (Pthreads).
  * **GPU Computing:** CUDA, OpenCL, HIP/ROCm, SYCL/oneAPI.
  * **Distributed Memory:** MPI, Hadoop, Spark.
  * *Rule of Thumb:* Align the programming model with the nature of the parallelism (e.g., CUDA for massively data-parallel kernels; OpenMP for shared-memory loops; MPI for multi-node/distributed scale).
* **Hardware & Compute Resources:**
  * Free choice of target machine: NCSU ARC, NCSU HPC (Hazel), local workstations, or dedicated GPU/accelerator platforms.
* **Team Structure:**
  * **Single-person (Solo):** Scopes, implements, tests, and analyzes the entire project individually.
  * **Two-person Team (Max 2):** Allowed; scope must be visibly larger (e.g., additional optimizations, multi-architecture comparisons, deeper analysis), with clearly defined task ownership for each member.

---

## 4. Deliverables, Timeline & Grading

| Deliverable | Description & Scope | Due Date (Fall 2026) | Weight |
| :--- | :--- | :--- | :--- |
| **Project Proposal** | 1 page covering problem statement, planned methodology, baseline plan, and target hardware. | Sep 16, 11:59 PM ET | Ungraded |
| **Oral Presentation** | **4-minute presentation + 30-second Q&A** delivered in class. | Nov 18 / Nov 23 / Nov 30 | **10%** |
| **Final Report + Code** | **Minimum 4 pages** (figures and tables included, references excluded) in standard ACM 2-column format + full source code. | Dec 7, 11:59 PM ET | **20%** |

---

## 5. Standard Project Workflow

```
1. Correct Serial Baseline + Accurate Timing
                     │
                     ▼
2. Profiling (Identify Hot Spots & Inefficiencies)
                     │
                     ▼
3. Parallelize & Optimize Bottlenecks
                     │
                     ▼
4. Verification (Validate Correctness Against Baseline Output)
                     │
                     ▼
5. Scaling Study (Sweep Cores/Threads/GPUs & Problem Sizes)
```

---

## 6. Final Report Specifications ("Show, Don't Claim")

* **Format:** Minimum 4 pages, ACM Primary Article 2-column format (LaTeX/Overleaf or Word).
* **Submission Package:** PDF report alongside well-documented source code submitted to Moodle.
* **Required Sections & Structure:**
  1. **Problem Statement & Motivation:** Mathematical background, real-world context, and rationale for parallel acceleration.
  2. **Baseline Implementation:** Description of the serial baseline, timing methodologies, and profiling results justifying parallel hotspots.
  3. **Parallel Architecture & Design:** Threading model, task/data distribution, memory hierarchy utilization, communication patterns, and synchronization.
  4. **Experimental Setup:** Machine specifications (CPU/GPU microarchitecture, memory, node count), software stack, compiler flags, and benchmark datasets.
  5. **Results & Scaling Analysis:**
     * Speedup equation: $S_p = \frac{T_1}{T_p}$
     * Strong scaling (fixed problem size, increasing workers $p$) and/or weak scaling (problem size scales proportionally with $p$).
     * Measured performance curves compared against ideal theoretical speedup.
  6. **Performance Discussion & Conclusion:** Amdahl's Law analysis, explanation of scaling ceilings (memory bandwidth, synchronization, communication overhead), and profiling-driven insights.

---

## 7. Common Pitfalls to Avoid

* **No Baseline:** Reporting speedup numbers without a timed, functional serial reference.
* **"Wrong Answers, Fast":** Focusing exclusively on execution speed while producing incorrect or non-deterministic results. Always verify parallel results against baseline output.
* **Single Data Point:** Benchmarking only a single core/thread configuration rather than performing parameter sweeps over worker counts ($p$) and problem sizes ($N$).
* **Over-Scoping:** Undertaking an overly complex architecture that cannot be completed or measured within the semester timeline.
* **Optimizing Blind:** Hand-tuning code based on guesswork without profiling data to pinpoint where execution time is actually spent.
* **Late Start:** Leaving hardware runs and benchmarking to the final week, leaving no margin for node queue delays, convergence issues, or debugging.
