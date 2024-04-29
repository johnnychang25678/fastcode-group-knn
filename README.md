# K-Nearest Neighbors (KNN) 

## Introduction

The milestone of the project is dedicated to examine approaches for enhancing the efficiency of a particular K-Nearest Neighbors (KNN) algorithm on both CPU and GPU platforms. Furthermore, it seeks to deploy and evaluate the impact of utilizing OpenMP optimization techniques, with the objective of reducing the execution time (measured in cycle counts). 
Please visit the associated baseline codebase by this link: https://github.com/luizgh/knn

## Final Design

### Profiling Insights and Initial Attempts

Our initial use of a profiling tool identified significant bottlenecks within the knn::run method, particularly in the computation of squared distances and the generation of final results. Initial efforts to parallelize the for-loop in the knn::run method did not yield any improvement. Profiling further highlighted that the sorting function was the primary contributor to extended execution times.

### Algorithm Optimization

After consulting with our professor and further analysis, we realized that sorting could be circumvented in the KNN algorithm, which would significantly reduce time complexity. By adopting a minimum heap approach, we were able to enhance the execution speed by a factor of two, effectively doubling the performance.

### Challenges with Parallel Implementations

Subsequent attempts to implement parallel versions of the algorithm using OpenMP and GPU technologies did not produce the expected speed-ups; in fact, these attempts slowed down the process. 

### Challenges with OpenMP Implementation
Code implementation:

```
KNNResults KNN::run(int k, DatasetPointer target) {
	// ...

	#pragma omp parallel shared(results, target, k)
	{
		int num_threads = omp_get_num_threads();
    	int thread_id = omp_get_thread_num();
	for(size_t targetExample = 0 + thread_id; targetExample < target->rows; targetExample += num_threads) {
// ...

```
The loop for(size_t targetExample = 0 + thread_id; targetExample < target->rows; targetExample += num_threads) assigns each thread a subset of the target dataset to process. The loop starts from the thread's unique ID (thread_id) and increments by the total number of threads (num_threads). The implementation ensures that each thread processes a distinct set of rows without overlapping, distributing the overall workload.

However, this method did not result in a speedup; instead, it slowed down the code by approximately 70% (detailed comparisons can be found in Section 3, Performance Plots). We concluded that the potential reason is that for the KNN algorithm, calculating distances or performing multiplications across all combinations of data involves frequent reading from and writing to memory locations that might be accessed by multiple threads. 

If each thread is processing a subset of data but those subsets reside close to each other in memory, the threads might invalidate each other's L1 caches. This not only leads to slowdowns due to waiting for data to be reloaded into the cache but also increases the bus traffic, which can degrade overall system performance.

### Challenges with CUDA Implementation
Code implementation can be found in `cuda/src/knn.cu`.

The CUDA kernel computeDistancesKernel is launched with a grid of threads where each thread computes the squared distance between a pair of training and target data points. The thread index tid is calculated as blockDim.x * blockIdx.x + threadIdx.x, uniquely identifying each thread’s task. This index maps to a specific combination of training and target rows, ensuring that each thread computes the distance for a unique pair of data points. The distances are stored in an array distancesDevice, which is accessed by tid to avoid any data overlap among threads, thus maximizing parallelism.

However, this method did not result in a speedup; instead, it slowed down the code by around 200%. According to the GNU gprof report, the use of CUDA resulted in extensive data movement operations. Although the computation of squared distances was accelerated, these additional operations slowed down the overall process. 

### Decision

Given the algorithm-optimized code is the only design with better performance than the baseline, we have decided to proceed with the single-threaded version as our eventual design choice. 

## Future Directions

Given our results, here are three approaches we would pursue to enhance performance, provided sufficient time:

1. Implement Advanced Data Structures:
  - Tree-based Structures: Utilize kd-trees for spatial data optimization, enhancing the efficiency of nearest-neighbor searches.
  - Hash Tables: Use hash tables for quicker data retrieval, significantly benefiting parallel computations by reducing the time complexity of access operations.
2. Minimize Data Movement in CUDA Implementation:
  - Data Partitioning: Reorganize the data to minimize movement, potentially through more localized computations or by re-evaluating how data is distributed between the host and the device.
  - Incremental Data Transfer: Transfer smaller data segments incrementally as needed, rather than moving the entire dataset at once. This approach is particularly advantageous if the GPU processes data in discrete stages, reducing idle time and improving throughput.
3. Optimize Memory Access Patterns:
  - Focus on enhancing the memory access pattern to boost cache utilization and decrease memory bandwidth bottlenecks. Optimized access patterns can lead to significant improvements in data processing efficiency and overall execution speed.

