# Compiler and flags
CXX = g++
CC = gcc
CXXFLAGS = -mfma -O2 -fopenmp -I.. -I. -I/home/docker/sdpa-dd/external/i/QD/include/
LDFLAGS = -L/home/docker/sdpa-dd/external/i/QD/lib -lqd

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

DD_AVX2_TEST_SRC = dd_avx2_test.cpp
DD_AVX2_TEST_OBJ = dd_avx2_test.o
DD_AVX2_TEST_BIN = dd_avx2_test
DD_AVX2_TEST_ASM = dd_avx2_test.s

RGEMM_KERNEL_SRC = Rgemm_NN_omp.cpp
RGEMM_KERNEL_OBJ = $(RGEMM_KERNEL_SRC:.cpp=.o)

Rgemm_test_SRC = Rgemm_test.cpp
Rgemm_test_OBJ = $(Rgemm_test_SRC:.cpp=.o)
Rgemm_test_BIN = Rgemm_test
Rgemm_test_ASM = Rgemm_test.s

all: $(Rgemm_test_BIN) $(Rgemm_test_ASM) $(DD_TEST_BIN) $(DD_TEST_ASM) $(DD_AVX2_TEST_BIN) $(DD_AVX2_TEST_ASM) $(TARGET_ORIG) $(ASM_ORIG) $(TARGET_MACRO) $(TARGET_BLOCKED)

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

$(DD_AVX2_TEST_OBJ): $(DD_AVX2_TEST_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(DD_AVX2_TEST_BIN): $(DD_AVX2_TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(Rgemm_test_OBJ): $(Rgemm_test_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(Rgemm_test_BIN): $(Rgemm_test_OBJ) $(RGEMM_KERNEL_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(ASM_ORIG): $(RGEMM_KERNEL_SRC)
	$(CXX) $(CXXFLAGS) -S -o $@ $<

$(DD_TEST_ASM): $(DD_TEST_SRC)
	$(CXX) $(CXXFLAGS) -S -o $@ $<

$(DD_AVX2_TEST_ASM): $(DD_AVX2_TEST_SRC)
	$(CXX) $(CXXFLAGS) -fverbose-asm -S -o $@ $<

$(Rgemm_test_ASM): $(Rgemm_test_SRC)
	$(CXX) $(CXXFLAGS) -S -o $@ $<

clean:
	rm -f $(OBJS_ORIG) $(OBJS_BLOCKED) $(OBJS_MACRO) $(DD_TEST_OBJ) $(Rgemm_test_OBJ) \
	      $(TARGET_ORIG) $(TARGET_BLOCKED) $(TARGET_MACRO) $(DD_TEST_BIN) $(Rgemm_test_BIN) \
	      $(ASM_ORIG) $(DD_TEST_ASM) $(Rgemm_test_ASM)
	rm -f $(DD_AVX2_TEST_OBJ) $(DD_AVX2_TEST_BIN) $(DD_AVX2_TEST_ASM)

.depend: $(SRCS_ORIG) $(SRCS_BLOCKED) $(SRCS_MACRO) $(DD_TEST_SRC) $(Rgemm_test_SRC)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend

-include .depend
