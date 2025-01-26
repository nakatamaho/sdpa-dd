/*
 * Copyright (c) 2025
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

#define __FMA(a, b, c) __builtin_fma((a), (b), (c))

#define TWO_SUM(a, b, s, e)                        \
do {                                               \
    (s) = (a) + (b);                               \
    double v = (s) - (a);                          \
    (e) = ((a) - ((s) - v)) + ((b) - v);           \
} while (0)

#define QUICK_TWO_SUM(a, b, s, e)                \
do {                                             \
    (s) = (a) + (b);                             \
    (e) = (b) - ((s) - (a));                     \
} while (0)

#define TWO_PROD_FMA(a, b, p, e)                \
do {                                            \
    (p) = (a) * (b);                            \
    (e) = __FMA((a), (b), -(p));                \
} while (0)

#define QUAD_ADD_IEEE(A, B, C)                                    \
do {                                                              \
    double s1, s11, s2, t1, t2;                                   \
    TWO_SUM((A).x[0], (B).x[0], s1, s2);                          \
    TWO_SUM((A).x[1], (B).x[1], t1, t2);                          \
    s2 += t1;                                                     \
    s11 = s1;/*workarounds the side effect of the macro expansion.*/\  
    QUICK_TWO_SUM(s11, s2, s1, s2);                               \
    s2 += t2;                                                     \
    QUICK_TWO_SUM(s1, s2, (C).x[0], (C).x[1]);                    \
} while (0)

#define QUAD_ADD_SLOPPY(A, B, C)                                 \
do {                                                             \
    double s, e;                                                 \
    TWO_SUM((A).x[0], (B).x[0], s, e);                           \
    e += (A).x[1] + (B).x[1];                                    \
    QUICK_TWO_SUM(s, e, (C).x[0], (C).x[1]);                     \
} while (0)

#define QUAD_MUL(A, B, C)                                        \
do {                                                             \
    double p1, p2;                                               \
    TWO_PROD_FMA((A).x[0], (B).x[0], p1, p2);                    \
    p2 += ((A).x[0] * (B).x[1]) + ((A).x[1] * (B).x[0]);         \
    QUICK_TWO_SUM(p1, p2, (C).x[0], (C).x[1]);                   \
} while (0)

#define QUAD_MUL_SLOPPY(A, B, C)                                  \
do {                                                              \
    double p, q, t, e;                                            \
    p = (A).x[0] * (B).x[1];                                      \
    q = (A).x[1] * (B).x[0];                                      \
    t = p + q;                                                    \
    (C).x[0] = __FMA((A).x[0], (B).x[0], t);                      \
    e = __FMA((A).x[0], (B).x[0], -(C).x[0]);                     \
    (C).x[1] = e + t;                                             \
} while (0)

#define QUAD_ADD_4_SLOPPY_AVX256(A, B, C)                        \
do {                                                             \
    /* AoSからSoA形式へのロード (インデックス順に注意) */         \
    __m256d a_hi = _mm256_setr_pd((A)[0].x[0], (A)[1].x[0],      \
                                 (A)[2].x[0], (A)[3].x[0]);      \
    __m256d a_lo = _mm256_setr_pd((A)[0].x[1], (A)[1].x[1],      \
                                 (A)[2].x[1], (A)[3].x[1]);      \
    __m256d b_hi = _mm256_setr_pd((B)[0].x[0], (B)[1].x[0],      \
                                 (B)[2].x[0], (B)[3].x[0]);      \
    __m256d b_lo = _mm256_setr_pd((B)[0].x[1], (B)[1].x[1],      \
                                 (B)[2].x[1], (B)[3].x[1]);      \
                                                                 \
    /* TWO_SUMのSIMD演算 */                                      \
    __m256d s = _mm256_add_pd(a_hi, b_hi);                       \
    __m256d v = _mm256_sub_pd(s, a_hi);                          \
    __m256d e = _mm256_add_pd(                                   \
        _mm256_sub_pd(a_hi, _mm256_sub_pd(s, v)),                \
        _mm256_sub_pd(b_hi, v)                                   \
    );                                                           \
                                                                 \
    /* ローパートの加算 */                                       \
    e = _mm256_add_pd(e, _mm256_add_pd(a_lo, b_lo));             \
                                                                 \
    /* QUICK_TWO_SUMのSIMD演算 */                                \
    __m256d s_new = _mm256_add_pd(s, e);                         \
    __m256d e_new = _mm256_sub_pd(e, _mm256_sub_pd(s_new, s));   \
                                                                 \
    /* 結果をAoS形式でストア */                                  \
    __m256d c0 = _mm256_unpacklo_pd(s_new, e_new);               \
    __m256d c1 = _mm256_unpackhi_pd(s_new, e_new);               \
                                                                 \
    _mm256_storeu_pd(&(C)[0].x[0], _mm256_permute2f128_pd(c0, c1, 0x20)); \
    _mm256_storeu_pd(&(C)[2].x[0], _mm256_permute2f128_pd(c0, c1, 0x31)); \
} while(0)
