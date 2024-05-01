#!/bin/bash

# generate data
cd data
python3 datasize_test.py
cd ..

# test baseline
cd baseline
make clean && make
baseline_result1=$(./bin/main run ../data/size_train1 ../data/size_test1 3)
baseline_result2=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
baseline_result3=$(./bin/main run ../data/size_train3 ../data/size_test3 3)
cd ..

# test optimized
cd optimized
make clean && make
optimized_result1=$(./bin/main run ../data/size_train1 ../data/size_test1 3)
optimized_result2=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
optimized_result3=$(./bin/main run ../data/size_train3 ../data/size_test3 3)
cd ..

# test openmp
cd openmp
make clean && make 
openmp_result1=$(./bin/main run ../data/size_train1 ../data/size_test1 3)
openmp_result2=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
openmp_result3=$(./bin/main run ../data/size_train3 ../data/size_test3 3)
cd ..

# test cuda
cd cuda
make clean && make 
cuda_result1=$(./bin/main run ../data/size_train1 ../data/size_test1 3)
cuda_result2=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
cuda_result3=$(./bin/main run ../data/size_train3 ../data/size_test3 3)
cd ..

# remove data
cd data
rm size*
cd ..

echo ""
echo "===================="
echo ""
# output results
echo "Baseline result 1 ~ 3:"
echo $baseline_result1
echo $baseline_result2
echo $baseline_result3
echo ""
echo "===================="
echo ""
echo "Optimized result 1 ~ 3:"
echo $optimized_result1
echo $optimized_result2
echo $optimized_result3
echo ""
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
echo $cuda_result1
echo $cuda_result2
echo $cuda_result3