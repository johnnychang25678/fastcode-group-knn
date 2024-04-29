CC=g++
CC_FLAGS=-std=c++11 -Wall -g -O3 -fopenmp

# File names
EXEC = bin/main
EXECOBJ = $(EXEC:bin/%=obj/%.o)
SOURCES = $(wildcard src/*.cpp)
INCLUDES = $(wildcard src/*.h)
OBJECTS = $(SOURCES:src/%.cpp=obj/%.o)

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

# generate gprof report
gprof: $(OBJECTS)
	$(CC) $(CC_FLAGS) -pg $(OBJECTS) -o $(EXEC)
	./bin/main run ../data/train2 ../data/test2 3
	gprof ./bin/main gmon.out > gprof_analysis.txt

# generate perf report
perf: $(OBJECTS)
	$(CC) $(CC_FLAGS) $(OBJECTS) -o $(EXEC)
	perf stat ./bin/main run ../data/train2 ../data/test2 3 > perf_analysis.txt 2>&1

# To remove generated files
clean:
	rm -f $(EXEC) $(OBJECTS) $(TESTTARGETS)
