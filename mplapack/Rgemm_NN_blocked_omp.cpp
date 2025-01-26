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

#include "dd_macro.h"

#define PREFETCH_DISTANCE 64

#define BLOCK_M 4
#define BLOCK_N 4
#define BLOCK_K 4

void Rgemm_block_2x2_kernel(mplapackint &x, mplapackint &y, mplapackint &z, const dd_real &alpha, const dd_real *A, mplapackint &lda, const dd_real *B, mplapackint &ldb, dd_real *C, mplapackint &ldc) {
    dd_real c00 = 0.0;
    dd_real c01 = 0.0;
    dd_real c10 = 0.0;
    dd_real c11 = 0.0;

    for (mplapackint k = 0; k < z; k++) {
        dd_real a0 = A[x + k * lda];     // A(x, k)
        dd_real a1 = A[x + 1 + k * lda]; // A(x+1, k)

        dd_real b0 = B[k + (y)*ldb];       // B(k, y)
        dd_real b1 = B[k + (y + 1) * ldb]; // B(k, y+1)

        c00 += a0 * b0;
        c01 += a0 * b1;
        c10 += a1 * b0;
        c11 += a1 * b1;
    }
    C[x + y * ldc] = C[x + y * ldc] + alpha * c00;
    C[x + (y + 1) * ldc] = C[x + (y + 1) * ldc] + alpha * c01;
    C[x + 1 + y * ldc] = C[x + 1 + y * ldc] + alpha * c10;
    C[x + 1 + (y + 1) * ldc] = C[x + 1 + (y + 1) * ldc] + alpha * c11;
}

void Rgemm_block_2x2_macro_kernel(mplapackint &x, mplapackint &y, mplapackint &z, const dd_real &alpha, const dd_real *A, mplapackint &lda, const dd_real *B, mplapackint &ldb, dd_real *C, mplapackint &ldc) {
    dd_real c00 = {0.0, 0.0};
    dd_real c01 = {0.0, 0.0};
    dd_real c10 = {0.0, 0.0};
    dd_real c11 = {0.0, 0.0};

    for (mplapackint k = 0; k < z; k++) {
        dd_real a0 = A[x + k * lda];     // A(x, k)
        dd_real a1 = A[x + 1 + k * lda]; // A(x+1, k)

        dd_real b0 = B[k + y * ldb];       // B(k, y)
        dd_real b1 = B[k + (y + 1) * ldb]; // B(k, y+1)

        dd_real temp0, temp1, temp2, temp3;

        QUAD_MUL_SLOPPY(a0, b0, temp0);
        QUAD_ADD_SLOPPY(c00, temp0, c00);
        QUAD_MUL_SLOPPY(a0, b1, temp1);
        QUAD_ADD_SLOPPY(c01, temp1, c01);
        QUAD_MUL_SLOPPY(a1, b0, temp2);
        QUAD_ADD_SLOPPY(c10, temp2, c10);
        QUAD_MUL_SLOPPY(a1, b1, temp3);
        QUAD_ADD_SLOPPY(c11, temp3, c11);
    }
    // alpha * c00
    dd_real alpha_c00, alpha_c01, alpha_c10, alpha_c11;
    QUAD_MUL_SLOPPY(alpha, c00, alpha_c00);
    QUAD_MUL_SLOPPY(alpha, c01, alpha_c01);
    QUAD_MUL_SLOPPY(alpha, c10, alpha_c10);
    QUAD_MUL_SLOPPY(alpha, c11, alpha_c11);

    QUAD_ADD_SLOPPY(C[x + y * ldc], alpha_c00, C[x + y * ldc]);
    QUAD_ADD_SLOPPY(C[x + (y + 1) * ldc], alpha_c01, C[x + (y + 1) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 1 + y * ldc], alpha_c10, C[x + 1 + y * ldc]);
    QUAD_ADD_SLOPPY(C[x + 1 + (y + 1) * ldc], alpha_c11, C[x + 1 + (y + 1) * ldc]);
}

void Rgemm_block_4x4_kernel(mplapackint &x, mplapackint &y, mplapackint &z, const dd_real &alpha, const dd_real *A, mplapackint &lda, const dd_real *B, mplapackint &ldb, dd_real *C, mplapackint &ldc) {
    dd_real c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
    dd_real c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
    dd_real c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
    dd_real c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;

    for (mplapackint k = 0; k < z; k++) {
        dd_real a0 = A[x + 0 + k * lda]; // A(x, k)
        dd_real a1 = A[x + 1 + k * lda]; // A(x+1, k)
        dd_real a2 = A[x + 2 + k * lda]; // A(x+2, k)
        dd_real a3 = A[x + 3 + k * lda]; // A(x+3, k)

        dd_real b0 = B[k + (y + 0) * ldb]; // B(k, y)
        dd_real b1 = B[k + (y + 1) * ldb]; // B(k, y+1)
        dd_real b2 = B[k + (y + 2) * ldb]; // B(k, y+2)
        dd_real b3 = B[k + (y + 3) * ldb]; // B(k, y+3)

        c00 += a0 * b0;
        c01 += a0 * b1;
        c02 += a0 * b2;
        c03 += a0 * b3;

        c10 += a1 * b0;
        c11 += a1 * b1;
        c12 += a1 * b2;
        c13 += a1 * b3;

        c20 += a2 * b0;
        c21 += a2 * b1;
        c22 += a2 * b2;
        c23 += a2 * b3;

        c30 += a3 * b0;
        c31 += a3 * b1;
        c32 += a3 * b2;
        c33 += a3 * b3;
    }
    C[x + 0 + (y + 0) * ldc] += alpha * c00;
    C[x + 0 + (y + 1) * ldc] += alpha * c01;
    C[x + 0 + (y + 2) * ldc] += alpha * c02;
    C[x + 0 + (y + 3) * ldc] += alpha * c03;

    C[x + 1 + (y + 0) * ldc] += alpha * c10;
    C[x + 1 + (y + 1) * ldc] += alpha * c11;
    C[x + 1 + (y + 2) * ldc] += alpha * c12;
    C[x + 1 + (y + 3) * ldc] += alpha * c13;

    C[x + 2 + (y + 0) * ldc] += alpha * c20;
    C[x + 2 + (y + 1) * ldc] += alpha * c21;
    C[x + 2 + (y + 2) * ldc] += alpha * c22;
    C[x + 2 + (y + 3) * ldc] += alpha * c23;

    C[x + 3 + (y + 0) * ldc] += alpha * c30;
    C[x + 3 + (y + 1) * ldc] += alpha * c31;
    C[x + 3 + (y + 2) * ldc] += alpha * c32;
    C[x + 3 + (y + 3) * ldc] += alpha * c33;
}

void Rgemm_block_4x4_macro_kernel(mplapackint &x, mplapackint &y, mplapackint &z, const dd_real &alpha, const dd_real *A, mplapackint &lda, const dd_real *B, mplapackint &ldb, dd_real *C, mplapackint &ldc) {
    dd_real c00 = {0.0, 0.0}, c01 = {0.0, 0.0}, c02 = {0.0, 0.0}, c03 = {0.0, 0.0};
    dd_real c10 = {0.0, 0.0}, c11 = {0.0, 0.0}, c12 = {0.0, 0.0}, c13 = {0.0, 0.0};
    dd_real c20 = {0.0, 0.0}, c21 = {0.0, 0.0}, c22 = {0.0, 0.0}, c23 = {0.0, 0.0};
    dd_real c30 = {0.0, 0.0}, c31 = {0.0, 0.0}, c32 = {0.0, 0.0}, c33 = {0.0, 0.0};

    for (mplapackint k = 0; k < z; k++) {
        dd_real a0 = A[x + 0 + k * lda]; // A(x, k)
        dd_real a1 = A[x + 1 + k * lda]; // A(x+1, k)
        dd_real a2 = A[x + 2 + k * lda]; // A(x+2, k)
        dd_real a3 = A[x + 3 + k * lda]; // A(x+3, k)

        dd_real b0 = B[k + (y + 0) * ldb]; // B(k, y)
        dd_real b1 = B[k + (y + 1) * ldb]; // B(k, y+1)
        dd_real b2 = B[k + (y + 2) * ldb]; // B(k, y+2)
        dd_real b3 = B[k + (y + 3) * ldb]; // B(k, y+3)

        dd_real temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12, temp13, temp14, temp15;

        QUAD_MUL_SLOPPY(a0, b0, temp0);
        QUAD_ADD_SLOPPY(c00, temp0, c00);
        QUAD_MUL_SLOPPY(a0, b1, temp1);
        QUAD_ADD_SLOPPY(c01, temp1, c01);
        QUAD_MUL_SLOPPY(a0, b2, temp2);
        QUAD_ADD_SLOPPY(c02, temp2, c02);
        QUAD_MUL_SLOPPY(a0, b3, temp3);
        QUAD_ADD_SLOPPY(c03, temp3, c03);
        QUAD_MUL_SLOPPY(a1, b0, temp4);
        QUAD_ADD_SLOPPY(c10, temp4, c10);
        QUAD_MUL_SLOPPY(a1, b1, temp5);
        QUAD_ADD_SLOPPY(c11, temp5, c11);
        QUAD_MUL_SLOPPY(a1, b2, temp6);
        QUAD_ADD_SLOPPY(c12, temp6, c12);
        QUAD_MUL_SLOPPY(a1, b3, temp7);
        QUAD_ADD_SLOPPY(c13, temp7, c13);
        QUAD_MUL_SLOPPY(a2, b0, temp8);
        QUAD_ADD_SLOPPY(c20, temp8, c20);
        QUAD_MUL_SLOPPY(a2, b1, temp9);
        QUAD_ADD_SLOPPY(c21, temp9, c21);
        QUAD_MUL_SLOPPY(a2, b2, temp10);
        QUAD_ADD_SLOPPY(c22, temp10, c22);
        QUAD_MUL_SLOPPY(a2, b3, temp11);
        QUAD_ADD_SLOPPY(c23, temp11, c23);
        QUAD_MUL_SLOPPY(a3, b0, temp12);
        QUAD_ADD_SLOPPY(c30, temp12, c30);
        QUAD_MUL_SLOPPY(a3, b1, temp13);
        QUAD_ADD_SLOPPY(c31, temp13, c31);
        QUAD_MUL_SLOPPY(a3, b2, temp14);
        QUAD_ADD_SLOPPY(c32, temp14, c32);
        QUAD_MUL_SLOPPY(a3, b3, temp15);
        QUAD_ADD_SLOPPY(c33, temp15, c33);
    }

    dd_real alpha_c00, alpha_c01, alpha_c02, alpha_c03;
    QUAD_MUL_SLOPPY(alpha, c00, alpha_c00);
    QUAD_MUL_SLOPPY(alpha, c01, alpha_c01);
    QUAD_MUL_SLOPPY(alpha, c02, alpha_c02);
    QUAD_MUL_SLOPPY(alpha, c03, alpha_c03);

    dd_real alpha_c10, alpha_c11, alpha_c12, alpha_c13;
    QUAD_MUL_SLOPPY(alpha, c10, alpha_c10);
    QUAD_MUL_SLOPPY(alpha, c11, alpha_c11);
    QUAD_MUL_SLOPPY(alpha, c12, alpha_c12);
    QUAD_MUL_SLOPPY(alpha, c13, alpha_c13);

    dd_real alpha_c20, alpha_c21, alpha_c22, alpha_c23;
    QUAD_MUL_SLOPPY(alpha, c20, alpha_c20);
    QUAD_MUL_SLOPPY(alpha, c21, alpha_c21);
    QUAD_MUL_SLOPPY(alpha, c22, alpha_c22);
    QUAD_MUL_SLOPPY(alpha, c23, alpha_c23);

    dd_real alpha_c30, alpha_c31, alpha_c32, alpha_c33;
    QUAD_MUL_SLOPPY(alpha, c30, alpha_c30);
    QUAD_MUL_SLOPPY(alpha, c31, alpha_c31);
    QUAD_MUL_SLOPPY(alpha, c32, alpha_c32);
    QUAD_MUL_SLOPPY(alpha, c33, alpha_c33);

    QUAD_ADD_SLOPPY(C[x + 0 + (y + 0) * ldc], alpha_c00, C[x + 0 + (y + 0) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 0 + (y + 1) * ldc], alpha_c01, C[x + 0 + (y + 1) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 0 + (y + 2) * ldc], alpha_c02, C[x + 0 + (y + 2) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 0 + (y + 3) * ldc], alpha_c03, C[x + 0 + (y + 3) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 1 + (y + 0) * ldc], alpha_c10, C[x + 1 + (y + 0) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 1 + (y + 1) * ldc], alpha_c11, C[x + 1 + (y + 1) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 1 + (y + 2) * ldc], alpha_c12, C[x + 1 + (y + 2) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 1 + (y + 3) * ldc], alpha_c13, C[x + 1 + (y + 3) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 2 + (y + 0) * ldc], alpha_c20, C[x + 2 + (y + 0) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 2 + (y + 1) * ldc], alpha_c21, C[x + 2 + (y + 1) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 2 + (y + 2) * ldc], alpha_c22, C[x + 2 + (y + 2) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 2 + (y + 3) * ldc], alpha_c23, C[x + 2 + (y + 3) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 3 + (y + 0) * ldc], alpha_c30, C[x + 3 + (y + 0) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 3 + (y + 1) * ldc], alpha_c31, C[x + 3 + (y + 1) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 3 + (y + 2) * ldc], alpha_c32, C[x + 3 + (y + 2) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 3 + (y + 3) * ldc], alpha_c33, C[x + 3 + (y + 3) * ldc]);
}
void Rgemm_NN_blocked_omp(mplapackint m, mplapackint n, mplapackint k, dd_real alpha, dd_real *A, mplapackint lda, dd_real *B, mplapackint ldb, dd_real beta, dd_real *C, mplapackint ldc) {
#pragma omp parallel for schedule(static)
    for (mplapackint j = 0; j < n; j++) {
        if (beta == 0.0) {
            for (mplapackint i = 0; i < m; i++) {
                C[i + j * ldc] = 0.0;
                if (i + PREFETCH_DISTANCE < m) {
                    __builtin_prefetch(&C[(i + PREFETCH_DISTANCE) + j * ldc], 1, 3);
                }
            }
        } else if (beta != 1.0) {
            dd_real new_val;
            for (mplapackint i = 0; i < m; i++) {
                // C[i + j * ldc] *= beta
                QUAD_MUL_SLOPPY(beta, C[i + j * ldc], new_val);
                C[i + j * ldc] = new_val;
                if (i + PREFETCH_DISTANCE < m) {
                    __builtin_prefetch(&C[(i + PREFETCH_DISTANCE) + j * ldc], 1, 3);
                }
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (mplapackint j0 = 0; j0 < n; j0 += BLOCK_N) {
        for (mplapackint i0 = 0; i0 < m; i0 += BLOCK_M) {
            for (mplapackint k0 = 0; k0 < k; k0 += BLOCK_K) {
                // Pointer redirecting by Nath et al
                // cf. https://doi.org/10.1007/978-3-642-19328-6_10
                mplapackint jb = (j0 + BLOCK_N <= n) ? BLOCK_N : (n - j0);
                mplapackint ib = (i0 + BLOCK_M <= m) ? BLOCK_M : (m - i0);
                mplapackint kb = (k0 + BLOCK_K <= k) ? BLOCK_K : (k - k0);
                dd_real *Ablk = &A[i0 + (size_t)k0 * lda];
                dd_real *Bblk = &B[k0 + (size_t)j0 * ldb];
                dd_real *Cblk = &C[i0 + (size_t)j0 * ldc];

#if (BLOCK_N == 2 && BLOCK_M == 2 && BLOCK_K == 2)
//                Rgemm_block_2x2_macro_kernel(ib, jb, kb, alpha, Ablk, lda, Bblk, ldb, Cblk, ldc);
                Rgemm_block_2x2_kernel(ib, jb, kb, alpha, Ablk, lda, Bblk, ldb, Cblk, ldc);
#elif (BLOCK_N == 4 && BLOCK_M == 4 && BLOCK_K == 4)
//                Rgemm_block_4x4_kernel(ib, jb, kb, alpha, Ablk, lda, Bblk, ldb, Cblk, ldc);
                Rgemm_block_4x4_macro_kernel(ib, jb, kb, alpha, Ablk, lda, Bblk, ldb, Cblk, ldc);
#else
#error "BLOCK_N, BLOCK_M, and BLOCK_K must all be either 2 or all be 4."
#endif
            }
        }
    }
}
