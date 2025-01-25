#include <mpblas_dd.h>

void Rgemm_ref(const char *transa, const char *transb, mplapackint m, mplapackint n, mplapackint k, dd_real alpha, dd_real *A, mplapackint lda, dd_real *B, mplapackint ldb, dd_real beta, dd_real *C, mplapackint ldc) {
    mplapackint nota, notb;
    mplapackint nrowa, ncola;
    mplapackint nrowb;
    mplapackint info;

    dd_real Zero = 0.0, One = 1.0;
    dd_real temp;

    nota = Mlsame_dd(transa, "N");
    notb = Mlsame_dd(transb, "N");

    if (nota) {
        nrowa = m;
        ncola = k;
    } else {
        nrowa = k;
        ncola = m;
    }
    if (notb) {
        nrowb = k;
    } else {
        nrowb = n;
    }

    // Test the input parameters.
    info = 0;
    if (!nota && (!Mlsame_dd(transa, "C")) && (!Mlsame_dd(transa, "T")))
        info = 1;
    else if (!notb && (!Mlsame_dd(transb, "C")) && (!Mlsame_dd(transb, "T")))
        info = 2;
    else if (m < 0)
        info = 3;
    else if (n < 0)
        info = 4;
    else if (k < 0)
        info = 5;
    else if (lda < std::max((mplapackint)1, nrowa))
        info = 8;
    else if (ldb < std::max((mplapackint)1, nrowb))
        info = 10;
    else if (ldc < std::max((mplapackint)1, m))
        info = 13;
    if (info != 0) {
        Mxerbla_dd("Rgemm ", info);
        return;
    }
    // Quick return if possible.
    if ((m == 0) || (n == 0) || (((alpha == Zero) || (k == 0)) && (beta == One)))
        return;

    // And when alpha == 0.0
    if (alpha == Zero) {
        if (beta == Zero) {
            for (mplapackint j = 0; j < n; j++) {
                for (mplapackint i = 0; i < m; i++) {
                    C[i + j * ldc] = Zero;
                }
            }
        } else {
            for (mplapackint j = 0; j < n; j++) {
                for (mplapackint i = 0; i < m; i++) {
                    C[i + j * ldc] = beta * C[i + j * ldc];
                }
            }
        }
        return;
    }
    // Start the operations.
    if (notb) {
        if (nota) {
            // Form C := alpha*A*B + beta*C.
            for (mplapackint j = 0; j < n; j++) {
                if (beta == Zero) {
                    for (mplapackint i = 0; i < m; i++) {
                        C[i + j * ldc] = Zero;
                    }
                } else if (beta != One) {
                    for (mplapackint i = 0; i < m; i++) {
                        C[i + j * ldc] = beta * C[i + j * ldc];
                    }
                }
                for (mplapackint l = 0; l < k; l++) {
                    if (B[l + j * ldb] != Zero) {
                        temp = alpha * B[l + j * ldb];
                        for (mplapackint i = 0; i < m; i++) {
                            C[i + j * ldc] = C[i + j * ldc] + temp * A[i + l * lda];
                        }
                    }
                }
            }
        } else {
            // Form  C := alpha*A'*B + beta*C.
            for (mplapackint j = 0; j < n; j++) {
                for (mplapackint i = 0; i < m; i++) {
                    temp = Zero;
                    for (mplapackint l = 0; l < k; l++) {
                        temp = temp + A[l + i * lda] * B[l + j * ldb];
                    }
                    if (beta == Zero)
                        C[i + j * ldc] = alpha * temp;
                    else
                        C[i + j * ldc] = alpha * temp + beta * C[i + j * ldc];
                }
            }
        }
    } else {
        if (nota) {
            // Form  C := alpha*A*B' + beta*C.
            for (mplapackint j = 0; j < n; j++) {
                if (beta == Zero) {
                    for (mplapackint i = 0; i < m; i++) {
                        C[i + j * ldc] = Zero;
                    }
                } else if (beta != One) {
                    for (mplapackint i = 0; i < m; i++) {
                        C[i + j * ldc] = beta * C[i + j * ldc];
                    }
                }
                for (mplapackint l = 0; l < k; l++) {
                    if (B[j + l * ldb] != Zero) {
                        temp = alpha * B[j + l * ldb];
                        for (mplapackint i = 0; i < m; i++) {
                            C[i + j * ldc] = C[i + j * ldc] + temp * A[i + l * lda];
                        }
                    }
                }
            }
        } else {
            // Form  C := alpha*A'*B' + beta*C.
            for (mplapackint j = 0; j < n; j++) {
                for (mplapackint i = 0; i < m; i++) {
                    temp = Zero;
                    for (mplapackint l = 0; l < k; l++) {
                        temp = temp + A[l + i * lda] * B[j + l * ldb];
                    }
                    if (beta == Zero)
                        C[i + j * ldc] = alpha * temp;
                    else
                        C[i + j * ldc] = alpha * temp + beta * C[i + j * ldc];
                }
            }
        }
    }
    return;
}
