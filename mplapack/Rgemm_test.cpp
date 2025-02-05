/*
 * Copyright (c) 2025
 *     Nakata, Maho
 *     All rights reserved.
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

#include <iostream>
#include <iomanip>
#include <random>
#include <mpblas_dd.h>

void Rgemm_NN_blocked_omp(mplapackint m, mplapackint n, mplapackint k, dd_real alpha, dd_real *A, mplapackint lda, dd_real *B, mplapackint ldb, dd_real beta, dd_real *C, mplapackint ldc);
void Rgemm_NN_omp(mplapackint m, mplapackint n, mplapackint k, dd_real alpha, dd_real *A, mplapackint lda, dd_real *B, mplapackint ldb, dd_real beta, dd_real *C, mplapackint ldc);

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

int main() {
    mplapackint m = 16;
    mplapackint n = 24;
    mplapackint k = 20;

    mplapackint lda = m + 4;
    mplapackint ldb = k + 8;
    mplapackint ldc = m + 10;

    dd_real *A = new dd_real[lda * k];
    dd_real *B = new dd_real[ldb * n];
    dd_real *C = new dd_real[ldc * n];
    dd_real *Cref = new dd_real[ldc * n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-9, 9);

    for (mplapackint j = 0; j < k; j++) {
        for (mplapackint i = 0; i < lda; i++) {
            A[i + j * lda] = -1000.0;
        }
    }
    for (mplapackint j = 0; j < n; j++) {
        for (mplapackint i = 0; i < ldb; i++) {
            B[i + j * ldb] = -1000.0;
        }
    }
    for (mplapackint j = 0; j < n; j++) {
        for (mplapackint i = 0; i < ldc; i++) {
            C[i + j * ldc] = -1000.0;
            Cref[i + j * ldc] = -1000.0;
        }
    }

    for (mplapackint j = 0; j < k; j++) {
        for (mplapackint i = 0; i < m; i++) {
            A[i + j * lda] = dis(gen);
        }
    }

    for (mplapackint j = 0; j < n; j++) {
        for (mplapackint i = 0; i < k; i++) {
            B[i + j * ldb] = dis(gen);
        }
    }

    for (mplapackint j = 0; j < n; j++) {
        for (mplapackint i = 0; i < m; i++) {
            C[i + j * ldc] = dis(gen);
            Cref[i + j * ldc] = C[i + j * ldc];
        }
    }

    dd_real alpha = dis(gen);
    dd_real beta = dis(gen);

    print_matrix_octave("A", A, m, k, lda);
    print_matrix_octave("B", B, k, n, ldb);
    print_matrix_octave("C", C, m, n, ldc);

    std::cout << "alpha = " << alpha << ";\n";
    std::cout << "beta  = " << beta << ";\n\n";

    std::cout << "Cnew = alpha * A * B + beta * C;\n\n";

    Rgemm_NN_blocked_omp(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    print_matrix_octave("Ccalc", C, m, n, ldc);
    std::cout << "Ccalc-Cnew" << std::endl;

    Rgemm_NN_omp(m, n, k, alpha, A, lda, B, ldb, beta, Cref, ldc);

    dd_real diff = 0.0;
    for (mplapackint j = 0; j < n; j++) {
        for (mplapackint i = 0; i < m; i++) {
            diff += abs(C[i + j * ldc].x[0] - Cref[i + j * ldc].x[0]);
        }
    }
    std::cout << "diff: " << diff << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] Cref;
    return 0;
}
