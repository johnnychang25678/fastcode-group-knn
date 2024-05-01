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

export OMP_NUM_THREADS=16

openmp_result1=$(./bin/main run ../data/size_train2 ../data/size_test2 3)
echo "k=3: $openmp_result1"

openmp_result2=$(./bin/main run ../data/size_train2 ../data/size_test2 10)
echo "k=10: $openmp_result2"

openmp_result3=$(./bin/main run ../data/size_train2 ../data/size_test2 25)
echo "k=25: $openmp_result3"

openmp_result4=$(./bin/main run ../data/size_train2 ../data/size_test2 50)
echo "k=50: $openmp_result4"

openmp_result5=$(./bin/main run ../data/size_train2 ../data/size_test2 100)
echo "k=100: $openmp_result5"

openmp_result6=$(./bin/main run ../data/size_train2 ../data/size_test2 200)
echo "k=200: $openmp_result6"

cd ..

# remove data
cd data
rm size*
cd ..