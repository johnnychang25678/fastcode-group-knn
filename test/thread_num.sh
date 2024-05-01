#!/bin/bash

# generate data
cd data
python3 datasize_test.py
cd ..

cd openmp
make clean && make

echo ""
echo "===================="
echo ""

export OMP_NUM_THREADS=1
openmp_result1=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
echo "1 thread: $openmp_result1"
export OMP_NUM_THREADS=2
openmp_result2=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
echo "2 threads: $openmp_result2"
export OMP_NUM_THREADS=4
openmp_result3=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
echo "4 threads: $openmp_result3"
export OMP_NUM_THREADS=8
openmp_result4=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
echo "8 threads: $openmp_result4"
export OMP_NUM_THREADS=16
openmp_result5=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
echo "16 threads: $openmp_result5"
export OMP_NUM_THREADS=32
openmp_result6=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
echo "32 threads: $openmp_result6"

cd ..

# remove data
cd data
rm size*
cd ..