# Source files and target
SRCS = Rgemm_NN_omp.cpp Rgemm_NN_omp_benchmark.cpp
OBJS = $(SRCS:.cpp=.o)
ASM = $(SRCS:.cpp=.s)
TARGET = Rgemm_NN_omp_benchmark

# Build target
all: $(TARGET)

# Compiler and flags
CXX = g++
CXXFLAGS = -mfma -O2 -fopenmp -I.. -I. -I/home/docker/sdpa-dd/external/i/QD/include/
#CXXFLAGS = -mno-fma -O2 -fopenmp -I.. -I. -I/home/docker/sdpa-dd/external/i/QD/include/
LDFLAGS = -L/home/docker/sdpa-dd/external/i/QD/lib -lqd

# Build the main target
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Generate assembly files
%.s: %.cpp
	$(CXX) $(CXXFLAGS) -S -o $@ $<

asm: $(ASM)

# Clean up
clean:
	rm -f $(OBJS) $(TARGET) $(ASM)

# Dependencies (optional, requires GCC/Clang)
depend:
	$(CXX) $(CXXFLAGS) -MM $(SRCS) > .depend

-include .depend
