/*
 * Copyright (c) 2010-2012
 *	Nakata, Maho
 * 	All rights reserved.
 *
 * $Id: Rgemm_NN.cpp,v 1.1 2010/12/28 06:13:53 nakatamaho Exp $
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */

#include <mpblas_dd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define MIN(a, b) ((a) < (b) ? (a) : (b))

void Rgemm_NN_blocked_omp(mplapackint m, mplapackint n, mplapackint k, dd_real alpha, dd_real *A, mplapackint lda, dd_real *B, mplapackint ldb, dd_real beta, dd_real *C, mplapackint ldc) {
    mplapackint i, j, l;
    dd_real temp;

    const mplapackint Bm = 16;
    const mplapackint Bn = 16;
    const mplapackint Bk = 16;

    const mplapackint Bn_init = 16;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(i, j)
#endif
    for (mplapackint jj = 0; jj < n; jj += Bn_init) {
        mplapackint j_end = MIN(jj + Bn_init, n);
        for (j = jj; j < j_end; j++) {
            if (beta == 0.0) {
                for (i = 0; i < m; i++) {
                    C[i + j * ldc] = 0.0;
                }
            } else if (beta != 1.0) {
                for (i = 0; i < m; i++) {
                    C[i + j * ldc] = beta * C[i + j * ldc];
                }
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(j, l, i, temp)
#endif
    for (mplapackint jj = 0; jj < n; jj += Bn) {
        for (mplapackint ll = 0; ll < k; ll += Bk) {
            for (mplapackint ii = 0; ii < m; ii += Bm) {
                for (j = jj; j < MIN(jj + Bn, n); j++) {
                    for (l = ll; l < MIN(ll + Bk, k); l++) {
                        // プリフェッチ: 次のループ反復で使用されるデータを事前に読み込む
                        if (l + 1 < MIN(ll + Bk, k)) {
                            __builtin_prefetch(&B[(l + 1) + j * ldb], 0, 3);
                        }
                        temp = alpha * B[l + j * ldb];
                        for (i = ii; i < MIN(ii + Bm, m); i++) {
                            // プリフェッチ: 次の行のデータを事前にキャッシュにロード
                            if (i + 1 < MIN(ii + Bm, m)) {
                                __builtin_prefetch(&A[(i + 1) + l * lda], 0, 3);
                                __builtin_prefetch(&C[(i + 1) + j * ldc], 1, 3);
                            }
                            C[i + j * ldc] += temp * A[i + l * lda];
                        }
                    }
                }
            }
        }
    }
}
