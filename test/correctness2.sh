#!/bin/bash

# generate data
echo "generating data..."
cd data
python3 feature_test.py
cd ..

cd optimized
make clean > /dev/null && make > /dev/null 2>&1
optimized_result2=$(./bin/main run ../data/feature_train2 ../data/feature_test2 3)
cd ..

echo "$optimized_result2" > /tmp/optimized_result2

# test openmp
cd openmp
make clean > /dev/null && make > /dev/null 2>&1
openmp_result2=$(./bin/main run ../data/feature_train2 ../data/feature_test2 3)
cd ..

echo "openmp:"
echo "$openmp_result2" > /tmp/openmp_result2
diff /tmp/optimized_result2 /tmp/openmp_result2
echo ""

# test cuda
cd cuda
make clean > /dev/null && make > /dev/null 2>&1
cuda_result2=$(./bin/main run ../data/feature_train2 ../data/feature_test2 3)
cd ..

echo "cuda:"
echo "$cuda_result2" > /tmp/cuda_result2
diff /tmp/optimized_result2 /tmp/cuda_result2
echo "===================="
echo ""

# remove data
cd data
rm feature_test1 feature_test2 feature_test3 feature_train1 feature_train2 feature_train3
cd ..
