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
#include <immintrin.h>
#include <stdio.h>

#define BLOCK_M 4
#define BLOCK_N 4
#define BLOCK_K 4

#define MR 4
#define NR 4

#define MC 8
#define NC 6
#define KC 4

#define PREFETCH_DISTANCE 64

#define MEM_ALIGN 64

static dd_real A_block[MC * KC] __attribute__((aligned(MEM_ALIGN)));
static dd_real B_block[NC * KC] __attribute__((aligned(MEM_ALIGN)));

#include <iomanip>
#include <iostream>

static void print_matrix_octave(const char *name, const dd_real *M, mplapackint m, mplapackint n, mplapackint ldm) {
    std::cout << name << " = [\n";
    for (mplapackint i = 0; i < m; i++) {
        std::cout << "    ";
        for (mplapackint j = 0; j < n; j++) {
            std::cout << std::setw(5) << M[i + j * ldm].x[0];
            if (j < n - 1) {
                std::cout << " ";
            }
        }
        if (i < m - 1) {
            std::cout << ";\n";
        }
    }
    std::cout << "\n];\n\n";
}

static inline void Rgemm_block_mc_nc_kc_kernel(mplapackint mc, mplapackint nc, mplapackint kc, const dd_real &alpha, dd_real *Ab, mplapackint ldab, dd_real *Bb, mplapackint ldbb, dd_real *Cb, mplapackint ldcb) {
    dd_real temp;
    for (mplapackint j = 0; j < nc; j++) {
        for (mplapackint l = 0; l < kc; l++) {
            temp = alpha * Bb[l + j * ldbb];
            for (mplapackint i = 0; i < mc; i++) {
                Cb[i + j * ldcb] += temp * Ab[i + l * ldab];
            }
        }
    }
    print_matrix_octave("Cb_new", Cb, mc, nc, ldcb);
}

void pack_A_block(dd_real *A, mplapackint lda, dd_real *A_block, mplapackint mc, mplapackint kc) {
    for (mplapackint i = 0; i < mc; i++) {
        for (mplapackint j = 0; j < kc; j++) {
            A_block[i + j * mc] = A[i + j * lda];
        }
    }
}

void pack_B_block(dd_real *B, mplapackint ldb, dd_real *B_block, mplapackint kc, mplapackint nc) {
    for (mplapackint j = 0; j < nc; j++) {
        for (mplapackint i = 0; i < kc; i++) {
            B_block[i + j * kc] = B[i + j * ldb];
        }
    }
}

void Rgemm_NN_blocked_omp(mplapackint M, mplapackint N, mplapackint K, dd_real alpha, dd_real *A, mplapackint lda, dd_real *B, mplapackint ldb, dd_real beta, dd_real *C, mplapackint ldc) {
    if (M % 4 != 0 || N % 4 != 0 || K % 4 != 0) {
        std::cerr << "Error: Matrix dimensions must be multiples of 4" << std::endl;
        exit(1);
    }
    //#pragma omp parallel for schedule(static)
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
    //#ifdef _OPENMP
    //#pragma omp parallel for collapse(2) schedule(static)
    //#endif
    for (mplapackint j = 0; j < N; j += NC) {
        const mplapackint nc = std::min((mplapackint)NC, N - j);
        for (mplapackint p = 0; p < K; p += KC) {
            const mplapackint kc = std::min((mplapackint)KC, K - p);
            pack_B_block(&B[p + j * ldb], ldb, B_block, kc, nc);
            for (mplapackint i = 0; i < M; i += MC) {
                const mplapackint mc = std::min((mplapackint)MC, M - i);
                pack_A_block(&A[i + p * lda], lda, A_block, mc, kc);
                Rgemm_block_mc_nc_kc_kernel(mc, nc, kc, alpha, A_block, mc, B_block, kc, &C[i + j * ldc], ldc);
            }
        }
    }
}
