# PSA02_OpenMP_kMeans_on_Arc
# CMDA 3634 PSA02 : OpenMP k‑means on ARC

**Author:** Jason Bruno Terceros  
**Course:** CMDA 3634 – Parallel Programming  
**Date:** February 2024 - March 2024

---

## 📌 Overview

This project parallelizes **k‑means clustering** (Lloyd’s algorithm) using **OpenMP** on Virginia Tech’s ARC (Advanced Research Computing) system. The dataset is the **MNIST test set** – 10,000 handwritten digit images, each with 784 dimensions (28×28 pixels).

We implement two parallel components:

1. **Farthest First algorithm** – initial cluster centroid selection (parallelized with OpenMP).
2. **Lloyd’s algorithm** – iterative cluster refinement (also parallelized with OpenMP).

The program is compiled with **static scheduling** (by default) and run on 1, 2, 4, 8, 16, and 32 threads. We measure strong scaling speedup and generate visualizations of the learned cluster centers.

> All work was done on ARC using Slurm batch scripts (`sbatch`).

---

## 🧩 Problem & Sequential Baseline

Given `num_points` points in `dim`‑dimensional space, k‑means partitions them into `k` clusters, each represented by its centroid. The algorithm:

1. **Choose initial centroids** (farthest first heuristic).
2. **Repeat** until convergence:
   - Assign each point to the nearest centroid.
   - Recompute each centroid as the mean of its assigned points.

The provided sequential code (`omp_kmeans.c`) performs both steps. Our task: introduce OpenMP parallelism to speed up the two most expensive parts:

- **Farthest first** – finding the point farthest from current centroids (`calc_arg_max`).
- **Lloyd’s iteration** – summing points per cluster (`calc_kmeans_next`).

---

## ⚙️ Implementation

### 1. Parallel Farthest First (`calc_arg_max`)

We parallelize the outer loop over all points. Each thread computes its local `thread_cost_sq` (maximum distance to the nearest of the current `m` centroids) and local `thread_arg_max` (index of that farthest point). A critical section combines the results.

```c
#pragma omp parallel default(none) shared(...)
{
    int thread_arg_max;
    double thread_cost_sq = 0;

    #pragma omp for schedule(static)   // or dynamic with -DDYNAMIC
    for (int i = 0; i < num_points; i++) {
        // compute min distance to any centroid
        // update thread_arg_max / thread_cost_sq if larger
    }

    #pragma omp critical
    {
        if (thread_cost_sq > cost_sq) {
            cost_sq = thread_cost_sq;
            arg_max = thread_arg_max;
        }
    }
}
```
### 2. Parallel k-means Update (`calc_kmeans_next`)

The Lloyd’s update has two parallel parts:
- **Point assignment** – each thread processes a subset of points, accumulates partial sums (`thread_kmeans_next`) and local cluster sizes (`thread_cluster_size`)
- **Critical reduction** – after the loop, threads add their partial sums and sizes to the global arrays

```c
#pragma omp parallel default(none) shared(...)
{
    int thread_cluster_size[k];
    double thread_kmeans_next[k*dim];
    vec_zero(thread_kmeans_next, k*dim);
    for (int i = 0; i < k; i++) thread_cluster_size[i] = 0;

    #pragma omp for schedule(static)
    for (int i = 0; i < num_points; i++) {
        int cluster = find_cluster(kmeans, data + i*dim, k, dim);
        vec_add(thread_kmeans_next + cluster*dim, data + i*dim, ...);
        thread_cluster_size[cluster]++;
    }

    #pragma omp critical
    {
        for (int i = 0; i < k; i++) cluster_size[i] += thread_cluster_size[i];
        vec_add(kmeans_next, thread_kmeans_next, kmeans_next, k*dim);
    }
}
```
> **Scheduling:** The code uses static scheduling by default. Dynamic scheduling can be enabled by compiling with `-DDYNAMIC`

---

## 📊 Results & Strong Scaling
We ran both parts on ARC using Slurm batch scripts. The tables below show wall‑clock times (seconds) and speedups relative to 1 thread.

### Farthest First Only (`m = 0`, i.e., only initial centroids)

| Threads | `k = 25` | Speedup | `k = 36` | Speedup |
|---------|----------|---------|----------|---------|
| 1 | 9.09 | 1.00 | 18.70 | 1.00 |
| 2 | 4.53 | 2.01 | 9.36 | 2.00 |
| 4 | 2.23 | 4.08 | 4.67 | 4.01 |
| 8 | 1.16 | 7.84 | 2.34 | 7.98 |
| 16 | 0.658 | 13.8 | 1.18 | 15.8 |
| 32 | 0.321 | 28.3 | 0.631 | 29.6 |

> Near-linear speedup: farthest first is embarrassingly parallel

### Full k‑means (`k = 16`, `m = 40` Lloyd iterations)

| Threads | Time (s) | Speedup |
|---------|----------|---------|
| 1 | 24.33 | 1.00 |
| 2 | 11.87 | 2.05 |
| 4 | 5.95 | 4.09 |
| 8 | 3.04 | 8.00 |
| 16 | 1.63 | 14.9 |
| 32 | 0.933 | 26.1 |

### Full k‑means (`k = 25`, `m = 55` Lloyd iterations)

| Threads | Time (s) | Speedup |
|---------|----------|---------|
| 1 | 58.58 | 1.00 |
| 2 | 29.38 | 1.99 |
| 4 | 14.60 | 4.01 |
| 8 | 6.99 | 8.38 |
| 16 | 3.96 | 14.8 |
| 32 | 2.08 | 28.1 |

> All runs show excellent strong scaling, especially for the farthest first stage (nearly perfect speedup). Lloyd’s algorithm also scales well, though the critical section introduces a small overhead.

---

## 🖼️ Visualizations
The learned cluster centers (as MNIST digit images) were generated using `mnist25.py` and `mnist16.py`. Below are examples:
- **Farthest first only** (`k = 25`, `m = 0`)
  - https://images/mnist_test_25_0.png
- **Full k‑means** (`k = 16`, `m = 40`)
  - https://images/mnist_test_16_40.png
- **Full k‑means** (`k = 25`, `m = 55`)
  - https://images/mnist_test_25_55.png

> The visualizations confirm that the parallel implementation produces the same cluster centers as the sequential version (visual agreement with provided examples).

---

## 🚀 How to Compile and Run on ARC

### 1. Clone the repository and load modules
  ```bash
  module purge
  module load gcc/12.2.0 openmpi
  ```

### 2. Compile (static scheduling, default)
  ```bash
  gcc -fopenmp -O2 -o omp_kmeans omp_kmeans.c vec.c
  ```

  For dynamic scheduling:
  ```bash
  gcc -fopenmp -O2 -DDYNAMIC -o omp_kmeans omp_kmeans.c vec.c
  ```

### 3. Run farthest first only (testing initial centroids)
  ```bash
  sbatch omp_kmeans.sh mnist_test.dat 25 0
  ```

### 4. Run full k-means
  ```bash
  sbatch omp_kmeans.sh mnist_test.dat 16 40
  cat omp_kmeans.out | python3 mnist16.py mnist_test_16_40.png
  ```

### 5. Strong scaling study
  ```bash
  sbatch omp_kmeans_timing.sh mnist_test.dat 25 0
  ```

The output contains pairs `(threads, time)` used to generate the scaling plots

---

📁 Repository Contents

CMDA3634_PSA02/
- src/main.tex
- images/
  - mnist_test_25_0.png
  - mnist_test_16_40.png
  - mnist_test_25_55.png
  - plot_25_0.png
  - plot_16_40.png
  - plot_25_55.png
- CMDA_3634_PSA02__OpenMP_k_means_on_ARC_Jason_BrunoTerceros.pdf
- README.md

---

## 🧠 Lessons Learned
- **Load balance** – The farthest first algorithm is perfectly balanced; static scheduling works well. For Lloyd’s algorithm, static scheduling is also adequate because each point’s work is roughly equal
- **Reduction patterns** – Using a critical section to combine per‑thread arrays is simple and effective when the reduction is relatively cheap compared to the parallel work
- **Strong scaling on ARC** – With up to 32 cores, we achieved speedups of 25–30×, demonstrating that OpenMP is efficient for data‑parallel k‑means


---

## 👤 Author
**Jason Bruno Terceros** – GitHub Profile
> Course: CMDA 3634 – Comp Sci Foundations
> Virginia Tech
