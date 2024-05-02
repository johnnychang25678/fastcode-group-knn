#include <iostream>
#include "knn.h"
#include "ReadDataset.h"
#include "dataset.h"
#include "Preprocessing.h"
#include <cstring>
#include <omp.h>

using namespace std;

const int nLabels = 10;

void runKnn(char *trainFile, char *testFile, int k) {
	DatasetPointer train = ReadDataset::read(trainFile, nLabels);
	DatasetPointer test = ReadDataset::read(testFile, nLabels);

	MatrixPointer meanData = MeanNormalize(train);

	KNN knn(train);

	ApplyMeanNormalization(test, meanData);

	double t0, t1;
	t0 = omp_get_wtime();
	KNNResults rawResults = knn.run(k, test);
	t1 = omp_get_wtime();
	
	double t_diff = t1 - t0;
	// printf("Time: %lf\n", t_diff);

	auto result = rawResults.getRawResults();
	for(size_t i = 0; i < result->rows; ++i) {
		cout << result->label(i);
	}
	cout << endl;
}

void findBestK(char *trainFile) {
	cout << "Reading train" <<endl;
	DatasetPointer data = ReadDataset::read(trainFile, nLabels);

	DatasetPointer train, valid;

	data->splitDataset(train, valid, 0.9);

	MatrixPointer meanData = MeanNormalize(train);
	ApplyMeanNormalization(valid, meanData);

	KNN knn(train);

	double bestSuccessRate = 0;
	int bestK = 0;

	for(int k=1; k<=10; k++) {
		printf("Trying K = %d ... ",k);
		KNNResults res = knn.run(k, valid);
		double currentSuccess = res.top1Result().successRate();
		if (currentSuccess > bestSuccessRate) {
			bestSuccessRate = currentSuccess;
			bestK = k;
		}
		printf("%lf\n", currentSuccess);
	}
	printf("Best K: %d. Success rate in validation set: %lf\n", bestK, bestSuccessRate);
}

void printUsageAndExit(char **argv);


int main(int argc, char **argv)
{
	if (argc != 3 && argc != 5) {
		printUsageAndExit(argv);
	}

	if (strcmp(argv[1], "run") == 0) {
		runKnn(argv[2], argv[3], atoi(argv[4]));
	}
	else if (strcmp(argv[1], "findbest") == 0) {
		findBestK(argv[2]);
	}
	else
		printUsageAndExit(argv);
}

void printUsageAndExit(char **argv) {
	printf("Usage:\n"
		"%s run <train dataset> <test dataset> <k> : run KNN\n"
		"%s findbest <train dataset> : Find K that minimizes error (1~10)\n",argv[0], argv[0]);
	exit(1);
}
