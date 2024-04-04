#include <iostream>
#include "knn.h"
#include "ReadDataset.h"
#include "dataset.h"
#include "Preprocessing.h"
#include <new>
#include <cstring>
#include "rdtsc.h"

using namespace std;

const int nLabels = 10;

void runKnn(char *trainFile, char *testFile, int k) {
	cout << "Reading train" <<endl;
	DatasetPointer train = ReadDataset::read(trainFile, nLabels);
	cout << "Reading test" <<endl;
	DatasetPointer test = ReadDataset::read(testFile, nLabels);

	MatrixPointer meanData = MeanNormalize(train);

	KNN knn(train);

	ApplyMeanNormalization(test, meanData);

	KNNResults rawResults = knn.run(k, test);
	SingleExecutionResults top1 = rawResults.top1Result();
	SingleExecutionResults top2 = rawResults.topXResult(2);
	SingleExecutionResults top3 = rawResults.topXResult(3);

	MatrixPointer confusionMatrix = rawResults.getConfusionMatrix();

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

	tsc_counter t0, t1;
	tsc_counter t2, t3;
	if (strcmp(argv[1], "run") == 0) {
		RDTSC(t0);
		runKnn(argv[2], argv[3], atoi(argv[4]));
		RDTSC(t1);
		cout << "runKnn time: " << COUNTER_DIFF(t1, t0, CYCLES) << endl;
	}
	else if (strcmp(argv[1], "findbest") == 0) {
		RDTSC(t2);
		findBestK(argv[2]);
		RDTSC(t3);
		cout << "findBestK time: " << COUNTER_DIFF(t3, t2, CYCLES) << endl;
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
