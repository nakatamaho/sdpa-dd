/*
 * Copyright (c) 2010-2025
 *	Nakata, Maho
 * 	All rights reserved.
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

#include <immintrin.h>
#include "dd_macro.h"

#define BLOCK_M 4
#define BLOCK_N 4
#define BLOCK_K 4

#define PREFETCH_DISTANCE 64

static inline void Rgemm_block_4x4_kernel(const dd_real &alpha, mplapackint k, dd_real *A, mplapackint lda, dd_real *B, mplapackint ldb, dd_real *C, mplapackint ldc) {
    dd_real c[4][4] = {0.0};
    dd_real b[4];
    dd_real *b_p0 = B;
    dd_real *b_p1 = B + ldb;
    dd_real *b_p2 = B + 2 * ldb;
    dd_real *b_p3 = B + 3 * ldb;

    for (int p = 0; p < k; ++p) {
        b[0] = *b_p0++;
        b[1] = *b_p1++;
        b[2] = *b_p2++;
        b[3] = *b_p3++;

        // c[0][0,1,2,3] += A[0][p] * b[p][0,1,2,3]
        QUAD_alpha_MAD_4_TYPE1_SLOPPY_AVX256(A[0], b, c[0]);

        // c[1][0,1,2,3] += A[1][p] * b[p][0,1,2,3]
        QUAD_alpha_MAD_4_TYPE1_SLOPPY_AVX256(A[1], b, c[1]);

        // c[2][0,1,2,3] += A[2][p] * b[p][0,1,2,3]
        QUAD_alpha_MAD_4_TYPE1_SLOPPY_AVX256(A[2], b, c[2]);

        // c[3][0,1,2,3] += A[3][p] * b[p][0,1,2,3]
        QUAD_alpha_MAD_4_TYPE1_SLOPPY_AVX256(A[3], b, c[3]);

        A += lda;
    }
/*
    b[0] = C[0 + 0 * ldc];
    b[1] = C[0 + 1 * ldc];
    b[2] = C[0 + 2 * ldc];
    b[3] = C[0 + 3 * ldc];
    QUAD_alpha_MAD_4_TYPE1_SLOPPY_AVX256(alpha, c[0], b);
    C[0 + 0 * ldc] = b[0];
    C[0 + 1 * ldc] = b[1];
    C[0 + 2 * ldc] = b[2];
    C[0 + 3 * ldc] = b[3];
*/
    C[0 + 0 * ldc] += alpha * c[0][0];
    C[0 + 1 * ldc] += alpha * c[0][1];
    C[0 + 2 * ldc] += alpha * c[0][2];
    C[0 + 3 * ldc] += alpha * c[0][3];

    C[1 + 0 * ldc] += alpha * c[1][0];
    C[1 + 1 * ldc] += alpha * c[1][1];
    C[1 + 2 * ldc] += alpha * c[1][2];
    C[1 + 3 * ldc] += alpha * c[1][3];

    C[2 + 0 * ldc] += alpha * c[2][0];
    C[2 + 1 * ldc] += alpha * c[2][1];
    C[2 + 2 * ldc] += alpha * c[2][2];
    C[2 + 3 * ldc] += alpha * c[2][3];

    C[3 + 0 * ldc] += alpha * c[3][0];
    C[3 + 1 * ldc] += alpha * c[3][1];
    C[3 + 2 * ldc] += alpha * c[3][2];
    C[3 + 3 * ldc] += alpha * c[3][3];
}

void Rgemm_NN_blocked_omp(mplapackint m, mplapackint n, mplapackint k, dd_real alpha, dd_real *A, mplapackint lda, dd_real *B, mplapackint ldb, dd_real beta, dd_real *C, mplapackint ldc) {
    if (m % 4 != 0 || n % 4 != 0 || k % 4 != 0) {
        std::cerr << "Error: Matrix dimensions must be multiples of 4" << std::endl;
        exit(1);
    }
#pragma omp parallel for schedule(static)
    for (mplapackint j = 0; j < n; ++j) {
        if (beta == 0.0) {
            for (mplapackint i = 0; i < m; ++i) {
                C[i + j * ldc] = 0.0;
                if (i + PREFETCH_DISTANCE < m) {
                    __builtin_prefetch(&C[(i + PREFETCH_DISTANCE) + j * ldc], 1, 3);
                }
            }
        } else if (beta != 1.0) {
            for (mplapackint i = 0; i < m; ++i) {
                C[i + j * ldc] = beta * C[i + j * ldc];
                if (i + PREFETCH_DISTANCE < m) {
                    __builtin_prefetch(&C[(i + PREFETCH_DISTANCE) + j * ldc], 1, 3);
                }
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (mplapackint j0 = 0; j0 < n; j0 += 4) {
        for (mplapackint i0 = 0; i0 < m; i0 += 4) {
            Rgemm_block_4x4_kernel(alpha, k, &A[i0 + 0 * lda], lda, &B[0 + j0 * ldb], ldb, &C[i0 + j0 * ldc], ldc);
        }
    }
}
