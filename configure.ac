dnl
dnl Configure for SDPA-DD 7.1.2
dnl

AC_INIT(sdpa-dd, 7.1.2)
AM_INIT_AUTOMAKE([subdir-objects])
AC_PROG_CC
AC_PROG_CXX
AC_CANONICAL_HOST

AC_ARG_WITH(system-spooles,
[  --with-system-spooles      Use spooles already on system],
WITH_SYSTEM_SPOOLES=$withval, WITH_SYSTEM_SPOOLES=no)
AM_CONDITIONAL(BUILD_SPOOLES, test x$WITH_SYSTEM_SPOOLES = xno)

AC_ARG_WITH(system-mplapack,
[  --with-system-mplapack      Use mplapack already on system],
WITH_SYSTEM_MPLAPACK=$withval, WITH_SYSTEM_MPLAPACK=no)
AM_CONDITIONAL(BUILD_MPLAPACK, test x$WITH_SYSTEM_MPLAPACK = xno)

AC_ARG_WITH(qd-includedir,
[  --with-qd-includedir    Gives qd include dir you use(e.g., /usr/local/include).],
QD_INCLUDEDIR=$withval
)

AC_ARG_WITH(qd-libdir,
[  --with-qd-libdir    Gives qd lib dir you use(e.g., /usr/local/lib).],
QD_LIBDIR=$withval
)

AC_ARG_WITH(mplapack-includedir,
[  --with-mplapack-includedir    Gives mplapack include dir you use(e.g., /usr/local/include).],
MPLAPACK_INCLUDEDIR=$withval
)

AC_ARG_WITH(mplapack-libdir,
[  --with-mplapack-libdir    Gives mplapack lib dir you use(e.g., /usr/local/lib).],
MPLAPACK_LIBDIR=$withval
)

AC_ARG_WITH(spooles-includedir,
[  --with-spooles-includedir    Gives spooles include dir you use(e.g., /usr/local/include/spooles).],
SPOOLES_INCLUDEDIR=$withval
)

AC_ARG_WITH(spooles-libdir,
[  --with-spooles-libdir    Gives spooles lib dir you use(e.g., /usr/local/lib).],
SPOOLES_LIBDIR=$withval
)

AC_ARG_ENABLE(metis,
[  --enable-metis    Build with metis],
[case "${enableval}" in
  yes) with_metis=yes ;;
  no)  with_metis=no  ;;
  *) AC_MSG_ERROR(bad value ${enableval} for --enable-metis) ;;
esac],[with_metis=no])
AM_CONDITIONAL(ENABLE_METIS, test x$with_metis = xyes)

AC_ARG_WITH(metis-includedir,
[  --with-metis-includedir    Gives metis include dir you use(e.g., /usr/local/include/metis).],
METIS_INCLUDEDIR=$withval
)

AC_ARG_WITH(metis-libdir,
[  --with-metis-libdir    Gives metis lib dir you use(e.g., /usr/local/lib).],
METIS_LIBDIR=$withval
)

AC_ARG_ENABLE(openmp,
[  --enable-openmp       Enables openmp
],WITH_OPENMP=yes)

AC_ARG_WITH(openmplib,
[  --with-openmplib    Gives openmplib],
OPENMPLIB=$withval
)

AC_ARG_WITH(openmpflag,
[  --with-openmpflag    Gives openmpflag],
OPENMPFLAG=$withval
)

AC_ARG_WITH(openmpthreads,
[  --with-openmpthread   Gives openmpthreads],
OPENMPTHREADS=$withval
)

AC_ARG_ENABLE(shared,
[  --enable-shared       Enables shared link to the libraries (slower)
],,)


dnl spooles and metis requires -lm when linking.
LIBS="-lm"

dnl How thread we link thread library esp. needed for spooles.
case "$host" in
        *-*-linux*|*-*-uclinux*)
            pthread_cflags="-D_REENTRANT"
            pthread_lib="-lpthread"
            ;;
        *-*-freebsd*|*-*-dragonfly*)
            pthread_cflags=""
            pthread_lib="-pthread"
            ;;
        *-apple-darwin*)
           enable_shared="yes"
        ;;
esac
LIBS="$LIBS $pthread_lib"
CXXFLAGS="$CXXFLAGS $pthread_cflags"
AC_SUBST(pthread_lib)

CFLAGS="$CFLAGS -O2 -funroll-all-loops"
CXXFLAGS="$CXXFLAGS -O2 -funroll-all-loops"
AC_SUBST(CFLAGS)
AC_SUBST(CXXFLAGS)

dnl some dummy settings
if test x$SPOOLES_INCLUDEDIR = x; then
SPOOLES_INCLUDEDIR="."
fi
if test x$SPOOLES_LIBDIR = x; then
SPOOLES_LIBDIR="."
fi

dnl

AC_MSG_CHECKING([whether to enable metis])
if test "$with_metis" = "no"; then
  AC_MSG_RESULT([no])
  ENABLE_METIS=NO
  AC_SUBST(ENABLE_METIS)
else
  AC_MSG_RESULT([yes])
  ENABLE_METIS=YES
  AC_SUBST(ENABLE_METIS)

  if test x$METIS_INCLUDEDIR = x; then
  METIS_INCLUDEDIR="."
  fi
  if test x$METIS_LIBDIR = x; then
  METIS_LIBDIR="."
  fi

  dnl Check for METIS include and library files
  HAVE_METIS_HEADER=""
  HAVE_METIS=""
  save_CPPFLAGS="$CPPFLAGS"
  CPPFLAGS="$CPPFLAGS -I$METIS_INCLUDEDIR"
  AC_CHECK_HEADER(metis.h, HAVE_METIS_HEADER=yes)
  CPPFLAGS=$save_CPPFLAGS
  if test x$HAVE_METIS_HEADER = xyes; then
    save_LDFLAGS="$LDFLAGS"
    LDFLAGS="$LDFLAGS -L$METIS_LIBDIR"
    AC_CHECK_LIB(metis, METIS_NodeND, [METIS_LIBS="-lmetis"; HAVE_METIS=yes])
    LDFLAGS=$save_LDFLAGS
  fi

  if test x$HAVE_METIS != xyes; then
    echo "Please install METIS"
    echo "METIS http://glaros.dtc.umn.edu/gkhome/views/metis/index.html"
    AC_MSG_ERROR([No METIS found])
  fi
  AC_SUBST(METIS_LIBDIR)
  AC_SUBST(METIS_INCLUDEDIR)
fi

if test x$WITH_SYSTEM_SPOOLES != xno; then
  dnl Check for SPOOLES include and library files
  HAVE_SPOOLES_HEADER=""
  HAVE_SPOOLES=""
  save_CPPFLAGS="$CPPFLAGS"
  CPPFLAGS="$CPPFLAGS -I$SPOOLES_INCLUDEDIR"
  AC_CHECK_HEADER(SPOOLES.h, HAVE_SPOOLES_HEADER=yes)
  CPPFLAGS=$save_CPPFLAGS

  if test x$HAVE_SPOOLES_HEADER = xyes; then
    save_LDFLAGS="$LDFLAGS"
    LDFLAGS="$LDFLAGS -L$SPOOLES_LIBDIR"
    AC_CHECK_LIB(spooles, InpMtx_init, [SPOOLES_LIB="-lspooles"; HAVE_SPOOLES=yes])
    LDFLAGS=$save_LDFLAGS
  fi

  if test x$HAVE_SPOOLES != xyes; then
    echo "Please install SPOOLES"
    echo "SPOOLES http://www.netlib.org/linalg/spooles/spooles.2.2.html"
    AC_MSG_ERROR([No SPOOLES found])
  fi
else
SPOOLES_LIBDIR=`pwd`/spooles/build
SPOOLES_INCLUDEDIR=`pwd`/spooles/build
fi
AC_SUBST(SPOOLES_LIBDIR)
AC_SUBST(SPOOLES_INCLUDEDIR)

dnl Check for QD include and library files
dnl some dummy settings
if test x$QD_INCLUDEDIR = x; then
QD_INCLUDEDIR="."
fi
if test x$QD_LIBDIR = x; then
QD_LIBDIR="."
fi

HAVE_QD_HEADER=""
HAVE_QD=""
AC_LANG_PUSH([C++])
save_CPPFLAGS="$CPPFLAGS"
CPPFLAGS="$CPPFLAGS -I$QD_INCLUDEDIR"
AC_CHECK_HEADER(qd/qd_real.h, HAVE_QD_HEADER=yes)
CPPFLAGS=$save_CPPFLAGS

if test x$HAVE_QD_HEADER = xyes; then
  save_LDFLAGS="$LDFLAGS"
  LDFLAGS="$LDFLAGS -L$QD_LIBDIR"
  AC_CHECK_LIB(qd, c_qd_sqrt, [QD_LIBS="-lqd"; HAVE_QD=yes])
  LDFLAGS=$save_LDFLAGS
fi

if test x$HAVE_QD != xyes; then
  echo "Please install QD"
  echo "Double-Double and Quad-Double Arithmetic http://www.cs.berkeley.edu/~yozo/"
  AC_MSG_ERROR([No QD found])
fi
AC_SUBST(QD_LIBDIR)
AC_SUBST(QD_INCLUDEDIR)

dnl Check for MPLAPACK include and library files
dnl some dummy settings
if test x$MPLAPACK_INCLUDEDIR = x; then
MPLAPACK_INCLUDEDIR="."
fi
if test x$MPLAPACK_LIBDIR = x; then
MPLAPACK_LIBDIR="."
fi

if test x$WITH_SYSTEM_MPLAPACK != xno; then
HAVE_MPLAPACK_HEADER=""
HAVE_MPLAPACK=""
save_CPPFLAGS="$CPPFLAGS"
CPPFLAGS="$CPPFLAGS -I$MPLAPACK_INCLUDEDIR"
AC_LANG_PUSH([C++])
AC_CHECK_HEADER(mblas_qd.h, HAVE_MPLAPACK_HEADER=yes)
CPPFLAGS=$save_CPPFLAGS

if test x$HAVE_MPLAPACK_HEADER = xyes; then
  save_LDFLAGS="$LDFLAGS"
  LDFLAGS="$LDFLAGS -L$MPLAPACK_LIBDIR"
  LIBS="$LDFLAGS -L$MPLAPACK_LIBDIR -lmlapack_dd -lmblas_dd -lqd";
 # AC_CHECK_FUNC(Rsyev, [HAVE_MPLAPACK=yes])
  HAVE_MPLAPACK=yes
  LDFLAGS=$save_LDFLAGS
fi

if test x$HAVE_MPLAPACK != xyes; then
  echo "Please install MPLAPACK (MBLAS/MLAPACK)"
  echo "MPLAPACK (MBLAS/MLAPACK) http://mplapack.cvs.sourceforge.net/"
  AC_MSG_ERROR([No MPLAPACK found])
fi
else
MPLAPACK_LIBDIR=`pwd`/mplapack
MPLAPACK_INCLUDEDIR=`pwd`/mplapack
fi
AM_CONDITIONAL(SYSTEM_MPLAPACK, test x$with_system_mplapack = xyes)
AC_SUBST(MPLAPACK_LIBDIR)
AC_SUBST(MPLAPACK_INCLUDEDIR)

dnl openmp
if test -n "$enable_openmp" && test "$enable_openmp" != "no" && test "$WITH_OPENMP" = "yes"; then
OPENMPFLAGS="-fopenmp -DENABLE_OPENMP -DNUM_OF_THREADS=$OPENMPTHREADS"
OPENMPLIB=-lgomp
fi
AC_SUBST(OPENMPLIB)
AC_SUBST(OPENMPFLAGS)

AM_CONDITIONAL(ENABLE_SHARED, test x$enable_shared = xyes)

AC_OUTPUT(Makefile)