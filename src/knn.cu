#include "knn.h"
#include <cassert>
#include <algorithm>
#include "dataset.h"
#include <iostream>
#include "debug.h"

__device__ double squaredDistance(const double* train, const double* target, int cols) {
    double sum = 0.0;
    for (int i = 0; i < cols; i++) {
        double diff = train[i] - target[i];
        sum += diff * diff;
    }
    return sum;
}

__global__ void computeDistancesKernel(const double* train, const double* target, double* distances, int trainRows, int targetRows, int cols) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= trainRows * targetRows) return;

    int trainIdx = tid / targetRows;
    int targetIdx = tid % targetRows;

    const double* trainPtr = &train[trainIdx * cols];
    const double* targetPtr = &target[targetIdx * cols];
    distances[tid] = squaredDistance(trainPtr, targetPtr, cols);
}


KNNResults KNN::run(int k, DatasetPointer target) {
	double* trainDevice;
    double* targetDevice;
    double* distancesDevice;

	int trainRows = data->rows;
	int targetRows = target->rows;
    int cols = target->cols;

	size_t trainSize = trainRows * cols * sizeof(double);
	size_t targetSize = targetRows * cols * sizeof(double);
	size_t distancesSize = targetRows * trainRows * sizeof(double);

	cudaMalloc(&trainDevice, trainSize);
	cudaMalloc(&targetDevice, targetSize);
	cudaMalloc(&distancesDevice, distancesSize);

	cudaMemcpy(trainDevice, data->data, trainSize, cudaMemcpyHostToDevice);
	cudaMemcpy(targetDevice, target->data, targetSize, cudaMemcpyHostToDevice);
	
	int blockSize = 256;
    int numBlocks = (trainRows * targetRows + blockSize - 1) / blockSize;
    computeDistancesKernel<<<numBlocks, blockSize>>>(trainDevice, targetDevice, distancesDevice, trainRows, targetRows, cols);

    double* distances = new double[trainRows * targetRows];
    cudaMemcpy(distances, distancesDevice, distancesSize, cudaMemcpyDeviceToHost);

    cudaFree(trainDevice);
    cudaFree(targetDevice);
    cudaFree(distancesDevice);

	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();

	for(size_t rowNum = 0; rowNum < target->rows; rowNum++) {
		std::pair<double, int> squaredDistances[trainRows];
		int startIdx = rowNum * trainRows;
		for(int i = 0; i < trainRows; ++i) {
			squaredDistances[i] = {distances[startIdx + i], i};
		}

		sort(squaredDistances, squaredDistances + trainRows);

		//count classes of nearest neighbors
		size_t nClasses = target->numLabels;
		int countClosestClasses[nClasses];
		for(size_t i = 0; i< nClasses; i++)
			 countClosestClasses[i] = 0;

		for (int i = 0; i < k; i++)
		{
			int currentClass = data->label(squaredDistances[i].second);
			countClosestClasses[currentClass]++;
		}

		//result: probability of class K for the example X
		for(size_t i = 0; i < nClasses; i++)
		{
			results->pos(rowNum, i) = ((double)countClosestClasses[i]) / k;
		}
	}

	//copy expected labels:
	for (size_t i = 0; i < target->rows; i++)
		results->label(i) = target->label(i);

	return KNNResults(results);
}
