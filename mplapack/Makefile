# Compiler and flags
CXX = g++
CXXFLAGS = -mfma -O2 -fopenmp -I.. -I. -I/home/docker/sdpa-dd/external/i/QD/include/
LDFLAGS = -L/home/docker/sdpa-dd/external/i/QD/lib -lqd

# Source files
SRCS_ORIG = Rgemm_ref.cpp Mlsame.cpp Mxerbla.cpp Rgemm_NN_omp.cpp Rgemm_NN_omp_benchmark.cpp
OBJS_ORIG = $(SRCS_ORIG:.cpp=.o)
TARGET_ORIG = Rgemm_NN_omp_benchmark
ASM_ORIG = Rgemm_NN_omp.s

SRCS_BLOCKED = Rgemm_NN_blocked_omp.cpp Rgemm_NN_omp.cpp Rgemm_NN_blocked_omp_benchmark.cpp
OBJS_BLOCKED = $(SRCS_BLOCKED:.cpp=.o)
TARGET_BLOCKED = Rgemm_NN_blocked_omp_benchmark

SRCS_MACRO = Rgemm_NN_macro_omp.cpp Rgemm_NN_omp.cpp Rgemm_NN_macro_omp_benchmark.cpp
OBJS_MACRO = $(SRCS_MACRO:.cpp=.o)
TARGET_MACRO = Rgemm_NN_macro_omp_benchmark

DD_TEST_SRC = dd_test.cpp
DD_TEST_OBJ = dd_test.o
DD_TEST_BIN = dd_test
DD_TEST_ASM = dd_test.s

# Targets
all: $(DD_TEST_BIN) $(DD_TEST_ASM) $(TARGET_ORIG) $(ASM_ORIG) $(TARGET_MACRO) $(TARGET_BLOCKED)

# Build rules
$(TARGET_ORIG): $(OBJS_ORIG)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_BLOCKED): $(OBJS_BLOCKED)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_MACRO): $(OBJS_MACRO)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(DD_TEST_OBJ): $(DD_TEST_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(DD_TEST_BIN): $(DD_TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Generate Rgemm_NN_omp.s
$(ASM_ORIG): Rgemm_NN_omp.cpp
	$(CXX) $(CXXFLAGS) -S -o $@ $<

# Generate dd_test.s
$(DD_TEST_ASM): $(DD_TEST_SRC)
	$(CXX) $(CXXFLAGS) -S -o $@ $<

# Clean and dependencies
clean:
	rm -f $(OBJS_ORIG) $(OBJS_BLOCKED) $(OBJS_MACRO) $(DD_TEST_OBJ) \
	      $(TARGET_ORIG) $(TARGET_BLOCKED) $(TARGET_MACRO) $(DD_TEST_BIN) \
	      $(ASM_ORIG) $(DD_TEST_ASM)

.depend: $(SRCS_ORIG) $(SRCS_BLOCKED) $(SRCS_MACRO) $(DD_TEST_SRC)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend

-include .depend
