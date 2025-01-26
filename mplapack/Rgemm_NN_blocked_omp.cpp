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

#define PREFETCH_DISTANCE 64

#define BLOCK_M 2
#define BLOCK_N 2
#define BLOCK_K 2

inline void Rgemm_block_2x2_kernel(mplapackint &x, mplapackint &y, mplapackint &z, const dd_real &alpha, const dd_real *A, mplapackint &lda, const dd_real *B, mplapackint &ldb, dd_real *C, mplapackint &ldc) {
    dd_real c[2][2] = {{{0.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {0.0, 0.0}}};
    for (mplapackint k = 0; k < z; k++) {
        dd_real a[2];
        a[0] = A[x + k * lda];     // A(x, k)
        a[1] = A[x + 1 + k * lda]; // A(x+1, k)
        dd_real b[2];
        b[0] = B[k + y * ldb];       // B(k, y)
        b[1] = B[k + (y + 1) * ldb]; // B(k, y+1)
        c[0][0] = c[0][0] + a[0] * b[0];
        c[0][1] = c[0][1] + a[0] * b[1];
        c[1][0] = c[1][0] + a[1] * b[0];
        c[1][1] = c[1][1] + a[1] * b[1];
    }
    C[x + y * ldc] = C[x + y * ldc] + alpha * c[0][0];
    C[x + (y + 1) * ldc] = C[x + (y + 1) * ldc] + alpha * c[0][1];
    C[x + 1 + y * ldc] = C[x + 1 + y * ldc] + alpha * c[1][0];
    C[x + 1 + (y + 1) * ldc] = C[x + 1 + (y + 1) * ldc] + alpha * c[1][1];
}

inline void Rgemm_block_2x2_macro_kernel(mplapackint &x, mplapackint &y, mplapackint &z, const dd_real &alpha, const dd_real *A, mplapackint &lda, const dd_real *B, mplapackint &ldb, dd_real *C, mplapackint &ldc) {
    dd_real c[2][2] = {{{0.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {0.0, 0.0}}};
    for (mplapackint k = 0; k < z; k++) {
        dd_real a[2];
        a[0] = A[x + k * lda];     // A(x, k)
        a[1] = A[x + 1 + k * lda]; // A(x+1, k)
        dd_real b[2];
        b[0] = B[k + y * ldb];       // B(k, y)
        b[1] = B[k + (y + 1) * ldb]; // B(k, y+1)
        dd_real temp[4];
        QUAD_MUL_SLOPPY(a[0], b[0], temp[0]);
        QUAD_ADD_SLOPPY(c[0][0], temp[0], c[0][0]);
        QUAD_MUL_SLOPPY(a[0], b[1], temp[1]);
        QUAD_ADD_SLOPPY(c[0][1], temp[1], c[0][1]);
        QUAD_MUL_SLOPPY(a[1], b[0], temp[2]);
        QUAD_ADD_SLOPPY(c[1][0], temp[2], c[1][0]);
        QUAD_MUL_SLOPPY(a[1], b[1], temp[3]);
        QUAD_ADD_SLOPPY(c[1][1], temp[3], c[1][1]);
    }
    dd_real alpha_c[2][2];
    QUAD_MUL_SLOPPY(alpha, c[0][0], alpha_c[0][0]);
    QUAD_MUL_SLOPPY(alpha, c[0][1], alpha_c[0][1]);
    QUAD_MUL_SLOPPY(alpha, c[1][0], alpha_c[1][0]);
    QUAD_MUL_SLOPPY(alpha, c[1][1], alpha_c[1][1]);
    QUAD_ADD_SLOPPY(C[x + y * ldc], alpha_c[0][0], C[x + y * ldc]);
    QUAD_ADD_SLOPPY(C[x + (y + 1) * ldc], alpha_c[0][1], C[x + (y + 1) * ldc]);
    QUAD_ADD_SLOPPY(C[x + 1 + y * ldc], alpha_c[1][0], C[x + 1 + y * ldc]);
    QUAD_ADD_SLOPPY(C[x + 1 + (y + 1) * ldc], alpha_c[1][1], C[x + 1 + (y + 1) * ldc]);
}

inline void Rgemm_block_4x4_kernel(mplapackint &x, mplapackint &y, mplapackint &z, const dd_real &alpha, const dd_real *A, mplapackint &lda, const dd_real *B, mplapackint &ldb, dd_real *C, mplapackint &ldc) {
    dd_real c[4][4] = {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}};
    for (mplapackint k = 0; k < z; k++) {
        dd_real a[4];
        a[0] = A[x + 0 + k * lda]; // A(x, k)
        a[1] = A[x + 1 + k * lda]; // A(x+1, k)
        a[2] = A[x + 2 + k * lda]; // A(x+2, k)
        a[3] = A[x + 3 + k * lda]; // A(x+3, k)
        dd_real b[4];
        b[0] = B[k + (y + 0) * ldb]; // B(k, y)
        b[1] = B[k + (y + 1) * ldb]; // B(k, y+1)
        b[2] = B[k + (y + 2) * ldb]; // B(k, y+2)
        b[3] = B[k + (y + 3) * ldb]; // B(k, y+3)
        c[0][0] += a[0] * b[0];
        c[0][1] += a[0] * b[1];
        c[0][2] += a[0] * b[2];
        c[0][3] += a[0] * b[3];
        c[1][0] += a[1] * b[0];
        c[1][1] += a[1] * b[1];
        c[1][2] += a[1] * b[2];
        c[1][3] += a[1] * b[3];
        c[2][0] += a[2] * b[0];
        c[2][1] += a[2] * b[1];
        c[2][2] += a[2] * b[2];
        c[2][3] += a[2] * b[3];
        c[3][0] += a[3] * b[0];
        c[3][1] += a[3] * b[1];
        c[3][2] += a[3] * b[2];
        c[3][3] += a[3] * b[3];
    }
    C[x + 0 + (y + 0) * ldc] += alpha * c[0][0];
    C[x + 0 + (y + 1) * ldc] += alpha * c[0][1];
    C[x + 0 + (y + 2) * ldc] += alpha * c[0][2];
    C[x + 0 + (y + 3) * ldc] += alpha * c[0][3];
    C[x + 1 + (y + 0) * ldc] += alpha * c[1][0];
    C[x + 1 + (y + 1) * ldc] += alpha * c[1][1];
    C[x + 1 + (y + 2) * ldc] += alpha * c[1][2];
    C[x + 1 + (y + 3) * ldc] += alpha * c[1][3];
    C[x + 2 + (y + 0) * ldc] += alpha * c[2][0];
    C[x + 2 + (y + 1) * ldc] += alpha * c[2][1];
    C[x + 2 + (y + 2) * ldc] += alpha * c[2][2];
    C[x + 2 + (y + 3) * ldc] += alpha * c[2][3];
    C[x + 3 + (y + 0) * ldc] += alpha * c[3][0];
    C[x + 3 + (y + 1) * ldc] += alpha * c[3][1];
    C[x + 3 + (y + 2) * ldc] += alpha * c[3][2];
    C[x + 3 + (y + 3) * ldc] += alpha * c[3][3];
}

inline void Rgemm_block_4x4_macro_kernel(mplapackint &x, mplapackint &y, mplapackint &z, const dd_real &alpha, const dd_real *A, mplapackint &lda, const dd_real *B, mplapackint &ldb, dd_real *C, mplapackint &ldc) {
    dd_real c[4][4] = {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}};
    for (mplapackint k = 0; k < z; k++) {
        dd_real a[4];
        a[0] = A[x + 0 + k * lda]; // A(x, k)
        a[1] = A[x + 1 + k * lda]; // A(x+1, k)
        a[2] = A[x + 2 + k * lda]; // A(x+2, k)
        a[3] = A[x + 3 + k * lda]; // A(x+3, k)
        dd_real b[4];
        b[0] = B[k + (y + 0) * ldb]; // B(k, y)
        b[1] = B[k + (y + 1) * ldb]; // B(k, y+1)
        b[2] = B[k + (y + 2) * ldb]; // B(k, y+2)
        b[3] = B[k + (y + 3) * ldb]; // B(k, y+3)
        dd_real temp[16];
        QUAD_MUL_SLOPPY(a[0], b[0], temp[0]);
        QUAD_ADD_SLOPPY(c[0][0], temp[0], c[0][0]);
        QUAD_MUL_SLOPPY(a[0], b[1], temp[1]);
        QUAD_ADD_SLOPPY(c[0][1], temp[1], c[0][1]);
        QUAD_MUL_SLOPPY(a[0], b[2], temp[2]);
        QUAD_ADD_SLOPPY(c[0][2], temp[2], c[0][2]);
        QUAD_MUL_SLOPPY(a[0], b[3], temp[3]);
        QUAD_ADD_SLOPPY(c[0][3], temp[3], c[0][3]);
        QUAD_MUL_SLOPPY(a[1], b[0], temp[4]);
        QUAD_ADD_SLOPPY(c[1][0], temp[4], c[1][0]);
        QUAD_MUL_SLOPPY(a[1], b[1], temp[5]);
        QUAD_ADD_SLOPPY(c[1][1], temp[5], c[1][1]);
        QUAD_MUL_SLOPPY(a[1], b[2], temp[6]);
        QUAD_ADD_SLOPPY(c[1][2], temp[6], c[1][2]);
        QUAD_MUL_SLOPPY(a[1], b[3], temp[7]);
        QUAD_ADD_SLOPPY(c[1][3], temp[7], c[1][3]);
        QUAD_MUL_SLOPPY(a[2], b[0], temp[8]);
        QUAD_ADD_SLOPPY(c[2][0], temp[8], c[2][0]);
        QUAD_MUL_SLOPPY(a[2], b[1], temp[9]);
        QUAD_ADD_SLOPPY(c[2][1], temp[9], c[2][1]);
        QUAD_MUL_SLOPPY(a[2], b[2], temp[10]);
        QUAD_ADD_SLOPPY(c[2][2], temp[10], c[2][2]);
        QUAD_MUL_SLOPPY(a[2], b[3], temp[11]);
        QUAD_ADD_SLOPPY(c[2][3], temp[11], c[2][3]);
        QUAD_MUL_SLOPPY(a[3], b[0], temp[12]);
        QUAD_ADD_SLOPPY(c[3][0], temp[12], c[3][0]);
        QUAD_MUL_SLOPPY(a[3], b[1], temp[13]);
        QUAD_ADD_SLOPPY(c[3][1], temp[13], c[3][1]);
        QUAD_MUL_SLOPPY(a[3], b[2], temp[14]);
        QUAD_ADD_SLOPPY(c[3][2], temp[14], c[3][2]);
        QUAD_MUL_SLOPPY(a[3], b[3], temp[15]);
        QUAD_ADD_SLOPPY(c[3][3], temp[15], c[3][3]);
    }

    dd_real alpha_c[4][4];
    QUAD_MUL_SLOPPY(alpha, c[0][0], alpha_c[0][0]);
    QUAD_MUL_SLOPPY(alpha, c[0][1], alpha_c[0][1]);
    QUAD_MUL_SLOPPY(alpha, c[0][2], alpha_c[0][2]);
    QUAD_MUL_SLOPPY(alpha, c[0][3], alpha_c[0][3]);
    QUAD_MUL_SLOPPY(alpha, c[1][0], alpha_c[1][0]);
    QUAD_MUL_SLOPPY(alpha, c[1][1], alpha_c[1][1]);
    QUAD_MUL_SLOPPY(alpha, c[1][2], alpha_c[1][2]);
    QUAD_MUL_SLOPPY(alpha, c[1][3], alpha_c[1][3]);
    QUAD_MUL_SLOPPY(alpha, c[2][0], alpha_c[2][0]);
    QUAD_MUL_SLOPPY(alpha, c[2][1], alpha_c[2][1]);
    QUAD_MUL_SLOPPY(alpha, c[2][2], alpha_c[2][2]);
    QUAD_MUL_SLOPPY(alpha, c[2][3], alpha_c[2][3]);
    QUAD_MUL_SLOPPY(alpha, c[3][0], alpha_c[3][0]);
    QUAD_MUL_SLOPPY(alpha, c[3][1], alpha_c[3][1]);
    QUAD_MUL_SLOPPY(alpha, c[3][2], alpha_c[3][2]);
    QUAD_MUL_SLOPPY(alpha, c[3][3], alpha_c[3][3]);
QUAD_ADD_4_SLOPPY_AVX256(alpha_c[0], &C[x + 0 + y * ldc], &C[x + 0 + y * ldc]);
QUAD_ADD_4_SLOPPY_AVX256(alpha_c[1], &C[x + 1 + y * ldc], &C[x + 1 + y * ldc]);
QUAD_ADD_4_SLOPPY_AVX256(alpha_c[2], &C[x + 2 + y * ldc], &C[x + 2 + y * ldc]);
QUAD_ADD_4_SLOPPY_AVX256(alpha_c[3], &C[x + 3 + y * ldc], &C[x + 3 + y * ldc]);
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
                Rgemm_block_2x2_kernel(ib, jb, kb, alpha, Ablk, lda, Bblk, ldb, Cblk, ldc);
//                Rgemm_block_2x2_macro_kernel(ib, jb, kb, alpha, Ablk, lda, Bblk, ldb, Cblk, ldc);
#elif (BLOCK_N == 4 && BLOCK_M == 4 && BLOCK_K == 4)
//              Rgemm_block_4x4_kernel(ib, jb, kb, alpha, Ablk, lda, Bblk, ldb, Cblk, ldc);
                Rgemm_block_4x4_macro_kernel(ib, jb, kb, alpha, Ablk, lda, Bblk, ldb, Cblk, ldc);
#else
#error "BLOCK_N, BLOCK_M, and BLOCK_K must all be either 2 or all be 4."
#endif
            }
        }
    }
}
