CC=g++
CC_FLAGS=-std=c++11 -Wall -g -O3
NVCC = nvcc
NVCC_FLAGS = -Xcompiler -fopenmp -std=c++11

# File names
EXEC = bin/main
EXECOBJ = $(EXEC:bin/%=obj/%.o)
SOURCES = $(wildcard src/*.cpp)
INCLUDES = $(wildcard src/*.h)
OBJECTS = $(SOURCES:src/%.cpp=obj/%.o)
CUDA_SOURCES = $(wildcard src/*.cu)
CUDA_OBJECTS = $(CUDA_SOURCES:src/%.cu=obj/%.o)

TESTSOURCES = $(wildcard test/*.cpp)
TESTOBJECTS = $(filter-out $(EXECOBJ), $(OBJECTS))
TESTTARGETS = $(TESTSOURCES:test/%.cpp=bin/%)

all: $(EXEC)

test: $(TESTTARGETS)

# Main target
$(EXEC): $(OBJECTS)
	$(CC) $(CC_FLAGS) $(OBJECTS) -o $(EXEC)

$(TESTTARGETS): bin/% : test/%.cpp $(TESTOBJECTS)
	$(CC) $(CC_FLAGS) -o $@ $< $(TESTOBJECTS) -Isrc/
#	./$@

# To obtain object files
$(OBJECTS): obj/%.o : src/%.cpp
	$(CC) -c $(CC_FLAGS) $< -o $@

$(CUDA_OBJECTS): obj/%.o : src/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# generate gprof report
gprof: $(OBJECTS)
<<<<<<<< HEAD:baseline/Makefile
	$(CC) $(CC_FLAGS) -pg $(OBJECTS) -o $(EXEC)
========
	$(NVCC) $(NVCC_FLAGS) -pg $(OBJECTS) $(CUDA_OBJECTS) -o $(EXEC)
>>>>>>>> cuda:cuda/Makefile
	./bin/main run ../data/train2 ../data/test2 3
	gprof ./bin/main gmon.out > gprof_analysis.txt

# generate perf report
<<<<<<<< HEAD:baseline/Makefile
perf: $(OBJECTS)
	$(CC) $(CC_FLAGS) $(OBJECTS) -o $(EXEC)
========
perf: $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) $(CUDA_OBJECTS) -o $(EXEC)
>>>>>>>> cuda:cuda/Makefile
	perf stat ./bin/main run ../data/train2 ../data/test2 3 > perf_analysis.txt 2>&1

# To remove generated files
clean:
<<<<<<<< HEAD:baseline/Makefile
	rm -f $(EXEC) $(OBJECTS) $(TESTTARGETS) *.out *.txt
========
	rm -f $(EXEC) $(OBJECTS) $(CUDA_OBJECTS) $(TESTTARGETS)
>>>>>>>> cuda:cuda/Makefile
