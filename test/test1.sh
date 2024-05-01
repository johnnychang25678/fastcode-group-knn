#!/bin/bash

# test baseline
cd baseline
make clean && make && baseline_result=$(./bin/main run ../data/train2 ../data/test2 3)
cd ..

# test optimized
cd optimized
make clean && make && optimized_result=$(./bin/main run ../data/train2 ../data/test2 3)
cd ..

# test openmp
cd openmp
make clean && make && openmp_result=$(./bin/main run ../data/train2 ../data/test2 3)
cd ..

# test cuda
cd cuda
make clean && make && cuda_result=$(./bin/main run ../data/train2 ../data/test2 3)
cd ..

# output results
echo "Baseline result:"
echo $baseline_result
echo "Optimized result:"
echo $optimized_result
echo "OpenMP result:"
echo $openmp_result
echo "CUDA result:"
echo $cuda_result