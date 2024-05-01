#include "knn.h"
#include <cassert>
#include <algorithm>
#include "dataset.h"
#include <iostream>
#include "debug.h"
#include <omp.h>
#include <queue>

__global__ void GetSquaredDistance(
	double* train, 
	double* test, 
	int numFeatures, 
	int numDatas, 
	int testId, 
	double* result
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= numFeatures) return;

	double sum = 0;
	double difference;
	for(int i = 0; i < numFeatures; i++) {
		difference = train[idx * numFeatures + i] - test[testId * numFeatures + i];
		sum += difference * difference;
	}
	result[testId * numDatas + idx] = sum;
}

KNNResults KNN::run(int k, DatasetPointer target) {
	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();

	double* trainHost = data->data;
	double* trainDevice;
	cudaMalloc(&trainDevice, data->rows * data->cols * sizeof(double));
	cudaMemcpy(trainDevice, trainHost, data->rows * data->cols * sizeof(double), cudaMemcpyHostToDevice);

	double* targetHost = target->data;
	double* targetDevice;
	cudaMalloc(&targetDevice, target->rows * target->cols * sizeof(double));
	cudaMemcpy(targetDevice, targetHost, target->rows * target->cols * sizeof(double), cudaMemcpyHostToDevice);

	double* squaredDistancesDevice;
	cudaMalloc(&squaredDistancesDevice, target->rows * data->rows * sizeof(double));

	#pragma omp parallel shared(results, target, k)
	{
		int num_threads = omp_get_num_threads();
    	int thread_id = omp_get_thread_num();
	for(size_t targetExample = 0 + thread_id; targetExample < target->rows; targetExample += num_threads) {
		GetSquaredDistance<<<(data->rows + 255) / 256, 256>>>(
			trainDevice, 
			targetDevice, 
			data->cols, // numFeatures
			data->rows, // numDatas
			targetExample, // testId
			squaredDistancesDevice
		);

		double* squaredDistancesHost = new double[data->rows];
		cudaMemcpy(squaredDistancesHost , squaredDistancesDevice + targetExample * data->rows, data->rows * sizeof(double), cudaMemcpyDeviceToHost);
		
		//squaredDistances: first is the distance; second is the trainExample row
		std::priority_queue<std::pair<double, int>> pq;
		for(int i = 0; i < data->rows; i++) {
			pq.push(std::make_pair(squaredDistancesHost[i], i));
		}
		
		//count classes of nearest neighbors
		size_t nClasses = target->numLabels;
		int countClosestClasses[nClasses];
		for(size_t i = 0; i< nClasses; i++)
			 countClosestClasses[i] = 0;

		for (int i = 0; i < k; i++)
		{
			auto topElement = pq.top(); pq.pop();
			int currentClass = data->label(topElement.second);
			countClosestClasses[currentClass]++;
		}

		//result: probability of class K for the example X
		for(size_t i = 0; i < nClasses; i++)
		{
			results->pos(targetExample, i) = ((double)countClosestClasses[i]) / k;
		}
	}
	}

	//copy expected labels:
	for (size_t i = 0; i < target->rows; i++)
		results->label(i) = target->label(i);

	return KNNResults(results);
}
