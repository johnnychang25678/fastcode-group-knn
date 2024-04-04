#include <iostream>
#include <cstdio>
#include "dataset.h"
#include "knn.h"
#include <cassert>
#include <cmath>
#include "testUtils.h"
#include "rdtsc.h"
#include "ReadDataset.h"
#include "Preprocessing.h"

int main(int argc, char **argv) 
{
	if (argc != 3) {
		cout << "Usage: " << argv[0] << " <trainFile> <testFile>" << endl;
		exit(1);
	}

	DatasetPointer train = ReadDataset::read(argv[1], 4);
	DatasetPointer test = ReadDataset::read(argv[2], 4);

	MatrixPointer meanData = MeanNormalize(train);
	KNN knn(train);
	ApplyMeanNormalization(test, meanData);

	KNNResults target = knn.run(10, test);

	tsc_counter start_time, end_time;

	RDTSC(start_time);
	SingleExecutionResults top1 = target.top1Result();
	RDTSC(end_time);
	cout << "top1Result time: " << COUNTER_DIFF(start_time, end_time, CYCLES) << endl;

	RDTSC(start_time);
	SingleExecutionResults top3 = target.topXResult(3);
	RDTSC(end_time);
	cout << "top3Result time: " << COUNTER_DIFF(start_time, end_time, CYCLES) << endl;

	RDTSC(start_time);
	SingleExecutionResults top10 = target.topXResult(10);
	RDTSC(end_time);
	cout << "top10Result time: " << COUNTER_DIFF(start_time, end_time, CYCLES) << endl;

	RDTSC(start_time);
	MatrixPointer confusionMatrix = target.getConfusionMatrix();
	RDTSC(end_time);
	cout << "getConfusionMatrix time: " << COUNTER_DIFF(start_time, end_time, CYCLES) << endl;

	return 0;
}
