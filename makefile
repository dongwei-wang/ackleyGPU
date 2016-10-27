BIN         		:= ackley

# CUDA toolkit path
CUDA_INSTALL_PATH 	:= /usr/local/cuda
NVCC 				:= $(CUDA_INSTALL_PATH)/bin/nvcc

# include path
INCD 				:= -I$(CUDA_INSTALL_PATH)/include
# lib path
LIBS 				:= -L$(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart


# compile flags for g++ and nvcc
CXXFLAGS  			:= -O3
CXXFLAGS 			:= -Wall -pedantic -std=c++0x -ggdb -DDEBUG
NVCCFLAGS 			:= -Xptxas="-v" -O3 \
					-gencode arch=compute_30,code=sm_30 \
					-gencode arch=compute_35,code=sm_35 \
					-gencode arch=compute_50,code=sm_50 \
					-gencode arch=compute_52,code=sm_52

# macro for files
CPP_SOURCES       	:= $(wildcard *.cpp)
CU_SOURCES        	:= ackley_kernel.cu
HEADERS           	:= $(wildcard *.h)
CPP_OBJS          	:= $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           	:= $(patsubst %.cu, %.o, $(CU_SOURCES))

# start to make
all: $(BIN)
$(BIN): $(CU_OBJS) $(CPP_OBJS)
	$(CXX) $(CXXFLAGS) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(CU_OBJS): $(CU_SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $(CU_OBJS) -c $(CU_SOURCES)

# clean unnecessary files
clean:
	rm -f $(BIN) *.o tags

cppecho:
	echo "$(CPP_SOURCES)"

headerecho:
	echo "$(HEADERS)"

cuecho:
	echo "$(CU_SOURCES)"
