# Dongwei Wang
# wdw828@gmail.com

BIN := demo

CC=g++
# CXXFLAGS=-Wall -pedantic -std=c++11 -O3
CXXFLAGS=-Wall -pedantic -std=c++0x  -ggdb -DDEBUG

OBJECTS=demo.o  Benchmarks.o F3.o

$(BIN): $(OBJECTS)
	$(CC) $(CXXFLAGS) -o demo $(OBJECTS)

demo.o: demo.cpp Header.h  Benchmarks.h F3.h
	$(CC) $(CXXFLAGS) -c demo.cpp

Benchmarks.o:  Benchmarks.h Benchmarks.cpp
	$(CC) $(CXXFLAGS) -c Benchmarks.cpp

F3.o: F3.h Benchmarks.h F3.cpp
	$(CC) $(CXXFLAGS) -c F3.cpp

.PHONY : clean
clean:
	rm -f $(BIN) $(OBJECTS)
