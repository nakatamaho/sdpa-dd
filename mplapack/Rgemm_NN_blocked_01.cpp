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

#include <iomanip>
#include <iostream>

static void print_matrix(const char *name, const dd_real *M, mplapackint m, mplapackint n, mplapackint ldm) {
    std::cout << name << " = [\n";
    for (mplapackint i = 0; i < m; i++) {
        std::cout << "      ";
        for (mplapackint j = 0; j < n; j++) {
            std::cout << std::setw(4) << M[i + j * ldm].x[0];
            if (j < n - 1) {
                std::cout << " ";
            }
        }
        if (i < m - 1) {
            std::cout << ";\n";
        }
    }
    std::cout << "\n];\n";
}

#define PREFETCH_DISTANCE 64

static inline void Rgemm_block_4x4_kernel(const dd_real &alpha, dd_real *Ab, mplapackint lda, dd_real *Bb, mplapackint ldb, dd_real *Cb, mplapackint ldc) {
    for (mplapackint ir = 0; ir < 4; ir++) {
        for (mplapackint jr = 0; jr < 4; jr++) {
            dd_real sum = 0.0;
            for (mplapackint kr = 0; kr < 4; kr++) {
                sum = sum + Ab[ir + kr * lda] * Bb[kr + jr * ldb];
            }
            Cb[ir + jr * ldc] += alpha * sum;
        }
    }
}

void Rgemm_NN_blocked_omp(mplapackint m, mplapackint n, mplapackint k, dd_real alpha, dd_real *A, mplapackint lda, dd_real *B, mplapackint ldb, dd_real beta, dd_real *C, mplapackint ldc) {
    if (m % 4 != 0 || n % 4 != 0 || k % 4 != 0) {
        std::cerr << "Error: Matrix dimensions must be multiples of 4" << std::endl;
        exit(1);
    }
//#pragma omp parallel for schedule(static)
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
    //#ifdef _OPENMP
    //#pragma omp parallel for collapse(2) schedule(static)
    //#endif
    for (mplapackint j = 0; j < n; j += 4) {
        for (mplapackint i = 0; i < m; i += 4) {
            for (mplapackint p = 0; p < k; p += 4) {
                Rgemm_block_4x4_kernel(alpha, &A[i + p * lda], lda, &B[p + j * ldb], ldb, &C[i + j * ldc], ldc);
            }
        }
    }
}
