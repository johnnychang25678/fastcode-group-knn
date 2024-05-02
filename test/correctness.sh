#!/bin/bash

# generate data
echo "generating data..."
cd data
python3 datasize_test.py
python3 feature_test.py
cd ..

# test baseline
cd baseline
make clean > /dev/null && make > /dev/null 2>&1
baseline_result1=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
baseline_result2=$(./bin/main run ../data/feature_train2 ../data/feature_test2 3)
cd ..

echo ""
echo "===================="
echo "baseline result 1: $baseline_result1" | cut -c 1-100
echo "baseline result 2: $baseline_result2" | cut -c 1-100
echo "===================="
echo ""

# test optimized
cd optimized
make clean > /dev/null && make > /dev/null 2>&1
optimized_result1=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
optimized_result2=$(./bin/main run ../data/feature_train2 ../data/feature_test2 3)
cd ..

echo ""
echo "===================="
if [ "$baseline_result1" = "$optimized_result1" ]; then
    echo "optimized answer 1 correct"
else
    echo "wrong answer:"
    echo "$optimized_result1"
fi
if [ "$baseline_result2" = "$optimized_result2" ]; then
    echo "optimized answer 2 correct"
else
    echo "wrong answer:"
    echo "$optimized_result2"
fi

# test openmp
cd openmp
make clean > /dev/null && make > /dev/null 2>&1
openmp_result1=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
openmp_result2=$(./bin/main run ../data/feature_train2 ../data/feature_test2 3)
cd ..
if [ "$baseline_result1" = "$openmp_result1" ]; then
    echo "openmp answer 1 correct"
else
    echo "wrong answer:"
    echo "$openmp_result1"
fi
if [ "$baseline_result2" = "$openmp_result2" ]; then
    echo "openmp answer 2 correct"
else
    echo "wrong answer:"
    echo "$openmp_result2"
fi

# test cuda
cd cuda
make clean > /dev/null && make > /dev/null 2>&1
cuda_result1=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
cuda_result2=$(./bin/main run ../data/feature_train2 ../data/feature_test2 3)
cd ..

if [ "$baseline_result1" = "$cuda_result1" ]; then
    echo "cuda answer 1 correct"
else
    echo "wrong answer:"
    echo "$cuda_result1"
fi
if [ "$baseline_result2" = "$cuda_result2" ]; then
    echo "cuda answer 2 correct"
else
    echo "wrong answer:"
    echo "$cuda_result2"
fi
echo "===================="
echo ""

# remove data
cd data
rm size*
rm feature_test1 feature_test2 feature_test3 feature_train1 feature_train2 feature_train3
cd ..
