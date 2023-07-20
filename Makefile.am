bin_PROGRAMS = sdpa_dd

if SYSTEM_MPLAPACK
sdpa_dd_SOURCES = sdpa_chordal.cpp sdpa_dataset.cpp sdpa_io.cpp \
sdpa_jordan.cpp sdpa_linear.cpp sdpa_main.cpp sdpa_newton.cpp sdpa_parts.cpp \
sdpa_struct.cpp sdpa_tool.cpp sdpa_chordal.h sdpa_dataset.h \
sdpa_include.h sdpa_io.h sdpa_jordan.h sdpa_linear.h sdpa_newton.h \
sdpa_parts.h sdpa_right.h sdpa_struct.h sdpa_tool.h
else
sdpa_dd_SOURCES = sdpa_chordal.cpp sdpa_dataset.cpp sdpa_io.cpp \
sdpa_jordan.cpp sdpa_linear.cpp sdpa_main.cpp sdpa_newton.cpp sdpa_parts.cpp \
sdpa_struct.cpp sdpa_tool.cpp sdpa_chordal.h sdpa_dataset.h \
sdpa_include.h sdpa_io.h sdpa_jordan.h sdpa_linear.h sdpa_newton.h \
sdpa_parts.h sdpa_right.h sdpa_struct.h sdpa_tool.h \
mplapack/Rtrsm.cpp mplapack/Rsyrk.cpp\
mplapack/Rtrmm.cpp mplapack/Rtrsv.cpp mplapack/iMlaenv.cpp\
mplapack/Rlamch.cpp mplapack/Rlascl.cpp mplapack/Rsytrd.cpp mplapack/Rsterf.cpp\
mplapack/Rorgtr.cpp mplapack/Rlatrd.cpp mplapack/Rsyr2k.cpp mplapack/Rsytd2.cpp \
mplapack/Rlanst.cpp mplapack/Rlae2.cpp mplapack/Rlapy2.cpp mplapack/Rlasrt.cpp\
mplapack/Rorgql.cpp mplapack/Rorgqr.cpp mplapack/Rsymv.cpp mplapack/Rlarfg.cpp\
mplapack/Rsyr2.cpp mplapack/Rlassq.cpp mplapack/Rorg2l.cpp mplapack/Rlarft.cpp\
mplapack/Rlarfb.cpp mplapack/Rorg2r.cpp mplapack/Rnrm2.cpp mplapack/Rlarf.cpp \
mplapack/Rger.cpp mplapack/Rpotrf.cpp mplapack/Mxerbla.cpp mplapack/Rpotf2.cpp \
mplapack/Mlsame.cpp mplapack/Rscal.cpp mplapack/Rgemv.cpp \
mplapack/Rtrmv.cpp mplapack/Rsteqr.cpp mplapack/Rlaset.cpp mplapack/Rlaev2.cpp \
mplapack/Rlasr.cpp mplapack/Rlartg.cpp mplapack/Rswap.cpp mplapack/Rsyev.cpp\
mplapack/Rlansy.cpp\
mplapack/Raxpy_omp.cpp\
mplapack/Rcopy_omp.cpp\
mplapack/Rdot_omp.cpp\
mplapack/Rgemm_omp.cpp\
mplapack/Rgemm_NN_omp.cpp\
mplapack/Rgemm_NT_omp.cpp\
mplapack/Rgemm_TN_omp.cpp\
mplapack/Rgemm_TT_omp.cpp
endif

AUTOMAKE_OPTIONS = subdir-objects

if ENABLE_SHARED
else
sdpa_dd_LDFLAGS = -static
endif

if SYSTEM_MPLAPACK
sdpa_dd_LDADD = $(pthread_lib) -L$(SPOOLES_LIBDIR) -lspooles -L$(QD_LIBDIR) -lqd $(OPENMPLIB) -L$(MPLAPACK_LIBDIR) -lmlapack_dd -lmblas_dd
sdpa_dd_CXXFLAGS= $(pthread_cflags) -I$(SPOOLES_INCLUDEDIR) -DUseMETIS=0 -I$(QD_INCLUDEDIR) $(OPENMPFLAGS) -I$(MPLAPACK_INCLUDEDIR)
else
sdpa_dd_LDADD = $(pthread_lib) -L$(SPOOLES_LIBDIR) -lspooles -L$(QD_LIBDIR) -lqd $(OPENMPLIB)
sdpa_dd_CXXFLAGS= $(pthread_cflags) -I$(SPOOLES_INCLUDEDIR) -DUseMETIS=0 -I$(QD_INCLUDEDIR) $(OPENMPFLAGS) -I./mplapack
endif

SUBDIRS = external

EXTRA_DIST = \
param.sdpa \
external/qd/download/*.tar.gz \
external/qd/Makefile.am \
external/qd/Makefile.in \
external/spooles/download/*.tgz \
external/spooles/patches/patch* \
external/spooles/Makefile.am \
external/spooles/Makefile.in
