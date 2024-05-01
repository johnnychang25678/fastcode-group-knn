#!/bin/bash

# generate data
cd data
python3 feature_test.py
cd ..

# test openmp
cd openmp
make clean && make 
openmp_result1=$(./bin/main run ../data/feature_train1 ../data/feature_test1 3)
openmp_result2=$(./bin/main run ../data/feature_train2 ../data/feature_test2 3)
openmp_result3=$(./bin/main run ../data/feature_train3 ../data/feature_test3 3)
cd ..

# test cuda
cd cuda
make clean && make 
feature_result1=$(./bin/main run ../data/feature_train1 ../data/feature_test1 3)
feature_result2=$(./bin/main run ../data/feature_train2 ../data/feature_test2 3)
feature_result3=$(./bin/main run ../data/feature_train3 ../data/feature_test3 3)
cd ..

# remove data
cd data
rm feature_test1 feature_test2 feature_test3 feature_train1 feature_train2 feature_train3
cd ..

echo "===================="
echo ""
echo "OpenMP result 1 ~ 3:"
echo $openmp_result1
echo $openmp_result2
echo $openmp_result3
echo ""
echo "===================="
echo ""
echo "CUDA result 1 ~ 3:"
echo $feature_result1
echo $feature_result2
echo $feature_result3