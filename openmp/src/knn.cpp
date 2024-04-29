#include "knn.h"
#include <cassert>
#include <algorithm>
#include "dataset.h"
#include <iostream>
#include "debug.h"
#include <omp.h>
<<<<<<<< HEAD:baseline/src/knn.cpp
========
#include <queue>
>>>>>>>> algo:optimized/src/knn.cpp

double GetSquaredDistance(DatasetPointer train, size_t trainExample, DatasetPointer target, size_t targetExample, int thread_id) {
	assert(train->cols == target->cols);
	double sum = 0;
	double difference;
	for(size_t i = 0; i < train->cols; i++) {
		int col = (thread_id + i) % train->cols;
		difference = train->pos(trainExample, col) - target->pos(targetExample, col);
		sum += difference * difference;
	}	return sum;
}

KNNResults KNN::run(int k, DatasetPointer target) {
	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();

<<<<<<<< HEAD:optimized/src/knn.cpp
<<<<<<<< HEAD:baseline/src/knn.cpp
	//squaredDistances: first is the distance; second is the trainExample row
	std::pair<double, int> squaredDistances[data->rows];
========
	#pragma omp parallel for default(none) shared(results, target, k)
>>>>>>>> algo:optimized/src/knn.cpp
	for(size_t targetExample = 0; targetExample < target->rows; targetExample++) {
========
	#pragma omp parallel shared(results, target, k)
	{
		int num_threads = omp_get_num_threads();
    	int thread_id = omp_get_thread_num();
	for(size_t targetExample = 0 + thread_id; targetExample < target->rows; targetExample += num_threads) {
>>>>>>>> algo-omp:openmp/src/knn.cpp

		#ifdef DEBUG_KNN
			if (targetExample % 100 == 0)
				DEBUGKNN("Target %lu of %lu\n", targetExample, target->rows);
<<<<<<<< HEAD:baseline/src/knn.cpp
		#endif

========
#endif
		std::priority_queue<std::pair<double, int>> pq;
>>>>>>>> algo:optimized/src/knn.cpp
		//Find distance to all examples in the training set
		for (size_t trainExample = 0; trainExample < data->rows; trainExample++) {
				//squaredDistances: first is the distance; second is the trainExample row
				std::pair<double, int> squaredDistances;
				squaredDistances.first = GetSquaredDistance(data, trainExample, target, targetExample, thread_id);
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
<<<<<<<< HEAD:baseline/src/knn.cpp
			int currentClass = data->label(squaredDistances[i].second);

========
			auto topElement = pq.top(); pq.pop();
			int currentClass = data->label(topElement.second);
>>>>>>>> algo:optimized/src/knn.cpp
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
