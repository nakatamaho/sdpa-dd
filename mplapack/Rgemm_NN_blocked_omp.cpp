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

static inline void Rgemm_block_4x4_kernel(const dd_real (&A_block)[4][4], const dd_real (&B_block)[4][4], dd_real (&C_block)[4][4]) {
    for (mplapackint ii = 0; ii < 4; ++ii) {
        for (mplapackint jj = 0; jj < 4; ++jj) {
            dd_real sum = C_block[ii][jj];
            for (mplapackint kk = 0; kk < 4; ++kk) {
                sum += A_block[ii][kk] * B_block[kk][jj];
            }
            C_block[ii][jj] = sum;
        }
    }
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
            dd_real C_block[4][4] = {{0}};

            for (mplapackint k0 = 0; k0 < k; k0 += 4) {
                dd_real A_block[4][4], B_block[4][4];

                for (mplapackint ii = 0; ii < 4; ++ii) {
                    for (mplapackint kk = 0; kk < 4; ++kk) {
                        A_block[ii][kk] = A[(i0 + ii) + (k0 + kk) * lda];
                    }
                }

                for (mplapackint kk = 0; kk < 4; ++kk) {
                    for (mplapackint jj = 0; jj < 4; ++jj) {
                        B_block[kk][jj] = B[(k0 + kk) + (j0 + jj) * ldb];
                    }
                }

                Rgemm_block_4x4_kernel(A_block, B_block, C_block);
            }
            for (mplapackint ii = 0; ii < 4; ++ii) {
                for (mplapackint jj = 0; jj < 4; ++jj) {
                    C[i0 + ii + (j0 + jj) * ldc] += alpha * C_block[ii][jj];
                }
            }
        }
    }
}
