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

	tsc_counter t0, t1;
	RDTSC(t0);
	SingleExecutionResults top1 = target.top1Result();
	RDTSC(t1);
	cout << "top1Result time: " << COUNTER_DIFF(t1, t0, CYCLES) << endl;

	tsc_counter t2, t3;
	RDTSC(t2);
	SingleExecutionResults top3 = target.topXResult(3);
	RDTSC(t3);
	cout << "top3Result time: " << COUNTER_DIFF(t3, t2, CYCLES) << endl;

	tsc_counter t4, t5;
	RDTSC(t4);
	SingleExecutionResults top10 = target.topXResult(10);
	RDTSC(t5);
	cout << "top10Result time: " << COUNTER_DIFF(t5, t4, CYCLES) << endl;

	tsc_counter t6, t7;
	RDTSC(t6);
	MatrixPointer confusionMatrix = target.getConfusionMatrix();
	RDTSC(t7);
	cout << "getConfusionMatrix time: " << COUNTER_DIFF(t7, t6, CYCLES) << endl;

	return 0;
}
