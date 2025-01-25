# Compiler and flags
CXX = g++
CXXFLAGS = -mfma -O2 -fopenmp -I.. -I. -I/home/docker/sdpa-dd/external/i/QD/include/
LDFLAGS = -L/home/docker/sdpa-dd/external/i/QD/lib -lqd

# Source files
SRCS_ORIG = Rgemm_ref.cpp Mlsame.cpp Mxerbla.cpp Rgemm_NN_omp.cpp Rgemm_NN_omp_benchmark.cpp
OBJS_ORIG = $(SRCS_ORIG:.cpp=.o)
ASM_ORIG = $(SRCS_ORIG:.cpp=.s)
TARGET_ORIG = Rgemm_NN_omp_benchmark
TARGET_ORIG_ASM = $(TARGET_ORIG).s

SRCS_BLOCKED = Rgemm_NN_blocked_omp.cpp Rgemm_NN_blocked_omp_benchmark.cpp
OBJS_BLOCKED = $(SRCS_BLOCKED:.cpp=.o)
ASM_BLOCKED = $(SRCS_BLOCKED:.cpp=.s)
TARGET_BLOCKED = Rgemm_NN_blocked_omp_benchmark
TARGET_BLOCKED_ASM = $(TARGET_BLOCKED).s

SRCS_MACRO = Rgemm_NN_macro_omp.cpp Rgemm_NN_macro_omp_benchmark.cpp
OBJS_MACRO = $(SRCS_MACRO:.cpp=.o)
ASM_MACRO = $(SRCS_MACRO:.cpp=.s)
TARGET_MACRO = Rgemm_NN_macro_omp_benchmark
TARGET_MACRO_ASM = $(TARGET_MACRO).s

DD_TEST_SRC = dd_test.cpp
DD_TEST_OBJ = dd_test.o
DD_TEST_BIN = dd_test
DD_TEST_ASM = $(DD_TEST_BIN).s

# Targets
all: $(TARGET_ORIG) $(TARGET_BLOCKED) $(TARGET_MACRO) $(DD_TEST_BIN) \
     $(TARGET_ORIG_ASM) $(TARGET_BLOCKED_ASM) $(TARGET_MACRO_ASM) $(DD_TEST_ASM)

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

# Assembly generation rules
$(TARGET_ORIG_ASM): $(SRCS_ORIG)
	$(CXX) $(CXXFLAGS) -S -o $@ $^

$(TARGET_BLOCKED_ASM): $(SRCS_BLOCKED)
	$(CXX) $(CXXFLAGS) -S -o $@ $^

$(TARGET_MACRO_ASM): $(SRCS_MACRO)
	$(CXX) $(CXXFLAGS) -S -o $@ $^

$(DD_TEST_ASM): $(DD_TEST_SRC)
	$(CXX) $(CXXFLAGS) -S -o $@ $<

# Clean and dependencies
clean:
	rm -f $(OBJS_ORIG) $(OBJS_BLOCKED) $(OBJS_MACRO) $(DD_TEST_OBJ) \
	      $(TARGET_ORIG) $(TARGET_BLOCKED) $(TARGET_MACRO) $(DD_TEST_BIN) \
	      $(TARGET_ORIG_ASM) $(TARGET_BLOCKED_ASM) $(TARGET_MACRO_ASM) $(DD_TEST_ASM)

.depend: $(SRCS_ORIG) $(SRCS_BLOCKED) $(SRCS_MACRO) $(DD_TEST_SRC)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend

-include .depend
