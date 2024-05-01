#include "knn.h"
#include <cassert>
#include <algorithm>
#include "dataset.h"
#include <iostream>
#include "debug.h"
#include <omp.h>
#include <queue>

double GetSquaredDistance(double* train, double* test, int numFeatures) {
	double sum = 0;
	double difference;
	for(int i = 0; i < numFeatures; i++) {
		difference = train[i] - test[i];
		sum += difference * difference;
	}
	return sum;
}

KNNResults KNN::run(int k, DatasetPointer target) {
	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();

	#pragma omp parallel shared(results, target, k)
	{
		int num_threads = omp_get_num_threads();
    	int thread_id = omp_get_thread_num();
	for(size_t targetExample = 0 + thread_id; targetExample < target->rows; targetExample += num_threads) {
		double* targetData = &target->data[targetExample * target->cols];
		double* trainData;
		
		//squaredDistances: first is the distance; second is the trainExample row
		std::pair<double, int> squaredDistances;
		std::priority_queue<std::pair<double, int>> pq;
		//Find distance to all examples in the training set
		for (size_t trainExample = 0; trainExample < data->rows; trainExample++) {
				trainData = &data->data[trainExample * data->cols];
				squaredDistances.first = GetSquaredDistance(trainData, targetData, target->cols);
				squaredDistances.second = trainExample;
				pq.push(squaredDistances);
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