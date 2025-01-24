/*
 * Copyright (c) 2010-2025
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

#include "dd_macro.h"

#define PREFETCH_DISTANCE 64

void Rgemm_NN_macro_omp(mplapackint m, mplapackint n, mplapackint k, dd_real alpha, dd_real *A, mplapackint lda, dd_real *B, mplapackint ldb, dd_real beta, dd_real *C, mplapackint ldc) {
    mplapackint i, j, l;
    dd_real temp, prod, new_val;

    // Form C := alpha*A*B + beta*C.
#pragma omp parallel for private(i, new_val) schedule(static)
    for (j = 0; j < n; j++) {
        if (beta == 0.0) {
            for (i = 0; i < m; i++) {
                C[i + j * ldc] = 0.0;
                if (i + PREFETCH_DISTANCE < m) {
                    __builtin_prefetch(&C[(i + PREFETCH_DISTANCE) + j * ldc], 1, 3);
                }
            }
        } else if (beta != 1.0) {
            for (i = 0; i < m; i++) {
                // C[i + j * ldc] = beta * C[i + j * ldc];
                QUAD_MUL_SLOPPY(beta, C[i + j * ldc], new_val);
                C[i + j * ldc] = new_val;
                if (i + PREFETCH_DISTANCE < m) {
                    __builtin_prefetch(&C[(i + PREFETCH_DISTANCE) + j * ldc], 1, 3);
                }
            }
        }
    }
// main loop
#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(temp, prod, new_val)
#endif
    for (j = 0; j < n; j++) {
        for (l = 0; l < k; l++) {
            // temp = alpha * B[l + j * ldb];
            QUAD_MUL(alpha, B[l + j * ldb], temp);
            if (l + PREFETCH_DISTANCE < k) {
                __builtin_prefetch(&B[(l + PREFETCH_DISTANCE) + j * ldb], 0, 3);
            }
            for (i = 0; i < m; i++) {
                // prod = temp * A[i + l * lda];
                QUAD_MUL_SLOPPY(temp, A[i + l * lda], prod);
                if (i + PREFETCH_DISTANCE < m) {
                    __builtin_prefetch(&A[(i + PREFETCH_DISTANCE) + l * lda], 0, 3);
                }
                // C[i + j * ldc] += prod;
                QUAD_ADD_SLOPPY(C[i + j * ldc], prod, new_val);
                C[i + j * ldc] = new_val;
            }
        }
    }
    return;
}
