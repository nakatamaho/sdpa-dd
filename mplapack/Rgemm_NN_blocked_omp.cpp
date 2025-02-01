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
#include <cstddef>

#include "dd_macro.h"
#include <immintrin.h>
#include <array>

#define MR 4
#define NR 4
#define MC MR * 64
#define NC NR * 64
#define KC 2000
#define PREFETCH_DISTANCE 64
#define MEM_ALIGN 64

static dd_real blockA_packed[MC * KC] __attribute__((aligned(MEM_ALIGN)));
static dd_real blockB_packed[NC * KC] __attribute__((aligned(MEM_ALIGN)));

static inline void Rgemm_block_4x4_kernel(const dd_real &alpha, dd_real *A, dd_real *B, dd_real *C, mplapackint k, mplapackint lda, mplapackint ldb, mplapackint ldc) {
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

void pack_panelB(dd_real *B, dd_real *blockB_packed, const mplapackint nr, const mplapackint kc, const mplapackint ldb) {
    for (mplapackint p = 0; p < kc; p++) {
        for (mplapackint j = 0; j < nr; j++) {
            *blockB_packed++ = B[p + j * ldb];
        }
        for (mplapackint j = nr; j < (mplapackint)NR; j++) {
            *blockB_packed++ = 0.0;
        }
    }
}

void pack_blockB(dd_real *B, dd_real *blockB_packed, const mplapackint nc, const mplapackint kc, const mplapackint ldb) {
    for (mplapackint j = 0; j < nc; j += NR) {
        const mplapackint nr = std::min((mplapackint)NR, nc - j);
        pack_panelB(&B[j * ldb], &blockB_packed[j * kc], nr, kc, ldb);
    }
}

void pack_panelA(dd_real *A, dd_real *blockA_packed, const mplapackint mr, const mplapackint kc, const mplapackint lda) {
    for (mplapackint p = 0; p < kc; p++) {
        for (mplapackint i = 0; i < mr; i++) {
            *blockA_packed++ = A[i + p * lda];
        }
        for (mplapackint i = mr; i < MR; i++) {
            *blockA_packed++ = 0.0;
        }
    }
}

void pack_blockA(dd_real *A, dd_real *blockA_packed, const mplapackint mc, const mplapackint kc, const mplapackint lda) {
    for (mplapackint i = 0; i < mc; i += MR) {
        const mplapackint mr = std::min((mplapackint)MR, mc - i);
        pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, lda);
    }
}

void Rgemm_NN_blocked_omp(mplapackint M, mplapackint N, mplapackint K, dd_real alpha, dd_real *A, mplapackint lda, dd_real *B, mplapackint ldb, dd_real beta, dd_real *C, mplapackint ldc) {
    if (M % 4 != 0 || N % 4 != 0 || K % 4 != 0) {
        std::cerr << "Error: Matrix dimensions must be multiples of 4" << std::endl;
        exit(1);
    }
#pragma omp parallel for schedule(static)
    for (mplapackint j = 0; j < N; ++j) {
        if (beta == 0.0) {
            for (mplapackint i = 0; i < M; ++i) {
                C[i + j * ldc] = 0.0;
                if (i + PREFETCH_DISTANCE < M) {
                    __builtin_prefetch(&C[(i + PREFETCH_DISTANCE) + j * ldc], 1, 3);
                }
            }
        } else if (beta != 1.0) {
            for (mplapackint i = 0; i < M; ++i) {
                C[i + j * ldc] = beta * C[i + j * ldc];
                if (i + PREFETCH_DISTANCE < M) {
                    __builtin_prefetch(&C[(i + PREFETCH_DISTANCE) + j * ldc], 1, 3);
                }
            }
        }
    }
    for (mplapackint j = 0; j < N; j += NC) {
        const int nc = std::min((mplapackint)NC, N - j);
        for (mplapackint p = 0; p < K; p += KC) {
            const int kc = std::min((mplapackint)KC, K - p);
            pack_blockB(&B[p + j * ldb], blockB_packed, kc, nc, ldb);
            for (mplapackint i = 0; i < M; i += MC) {
                const int mc = std::min((mplapackint)MC, M - i);
                pack_blockA(&A[i + p * lda], blockA_packed, mc, kc, lda);
                for (mplapackint jr = 0; jr < nc; jr += NR) {
                    const int nr = std::min((mplapackint)NR, nc - jr);
                    for (mplapackint ir = 0; ir < mc; ir += MR) {
                        const int mr = std::min((mplapackint)MR, mc - ir);
                        Rgemm_block_4x4_kernel(alpha, &blockA_packed[ir * kc], &blockB_packed[jr * kc], &C[(i + ir) + (j + jr) * ldc], mr, nr, kc, ldc);
                    }
                }
            }
        }
    }
}
