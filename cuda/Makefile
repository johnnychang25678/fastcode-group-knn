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
	$(NVCC) $(NVCC_FLAGS) -pg $(OBJECTS) $(CUDA_OBJECTS) -o $(EXEC)
	./bin/main run ../data/train2 ../data/test2 3
	gprof ./bin/main gmon.out > gprof_analysis.txt

# generate perf report
perf: $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) $(CUDA_OBJECTS) -o $(EXEC)
	perf stat ./bin/main run ../data/train2 ../data/test2 3 > perf_analysis.txt 2>&1

# generate nvprof report
nvprof: $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) $(CUDA_OBJECTS) -o $(EXEC)
	nvprof ./bin/main run ../data/train2 ../data/test2 3 > nvprof_analysis.txt 2>&1

# To remove generated files
clean:
	rm -f $(EXEC) $(OBJECTS) $(CUDA_OBJECTS) $(TESTTARGETS)
