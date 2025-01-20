# Source files for original benchmark
SRCS_ORIG = Rgemm_NN_omp.cpp Rgemm_NN_omp_benchmark.cpp
OBJS_ORIG = $(SRCS_ORIG:.cpp=.o)
ASM_ORIG = $(SRCS_ORIG:.cpp=.s)
TARGET_ORIG = Rgemm_NN_omp_benchmark

# Source files for blocked benchmark
SRCS_BLOCKED = Rgemm_NN_blocked_omp.cpp Rgemm_NN_blocked_omp_benchmark.cpp
OBJS_BLOCKED = $(SRCS_BLOCKED:.cpp=.o)
ASM_BLOCKED = $(SRCS_BLOCKED:.cpp=.s)
TARGET_BLOCKED = Rgemm_NN_blocked_omp_benchmark

# Build targets
all: $(TARGET_ORIG) $(TARGET_BLOCKED)

# Compiler and flags
CXX = g++
CXXFLAGS = -mfma -O2 -fopenmp -I.. -I. -I/home/docker/sdpa-dd/external/i/QD/include/
#CXXFLAGS = -mno-fma -O2 -fopenmp -I.. -I. -I/home/docker/sdpa-dd/external/i/QD/include/
LDFLAGS = -L/home/docker/sdpa-dd/external/i/QD/lib -lqd

# Build original benchmark executable
$(TARGET_ORIG): $(OBJS_ORIG)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Build blocked benchmark executable
$(TARGET_BLOCKED): $(OBJS_BLOCKED)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Generating assembly files
%.s: %.cpp
	$(CXX) $(CXXFLAGS) -S -o $@ $<

asm: $(ASM_ORIG) $(ASM_BLOCKED)

# Clean up
clean:
	rm -f $(OBJS_ORIG) $(OBJS_BLOCKED) $(TARGET_ORIG) $(TARGET_BLOCKED) $(ASM_ORIG) $(ASM_BLOCKED)

# Dependencies (optional, requires GCC/Clang)
depend:
	$(CXX) $(CXXFLAGS) -MM $(SRCS_ORIG) $(SRCS_BLOCKED) > .depend

-include .depend
