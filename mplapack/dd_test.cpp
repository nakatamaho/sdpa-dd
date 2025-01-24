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

#include <iostream>
#include <random>
#include <qd/dd_real.h>

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
    double s1, s2, t1, t2;                                        \
    TWO_SUM((A).x[0], (B).x[0], s1, s2);                          \
    TWO_SUM((A).x[1], (B).x[1], t1, t2);                          \
    s2 += t1;                                                     \
    QUICK_TWO_SUM(s1, s2, s1, s2);                                \
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

int main() {
    using namespace std;
    using namespace qd;

    cout.precision(33);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    dd_real a_val, b_val;
    a_val.x[0] = dist(mt);
    a_val.x[1] = dist(mt) * 1e-16;
    b_val.x[0] = dist(mt);
    b_val.x[1] = dist(mt) * 1e-16;

    cout << "a_val = " << a_val << endl;
    cout << "b_val = " << b_val << endl;

    cout << "### addition and multiplication test ###" << endl;

    dd_real sum = a_val + b_val;
    cout << "a_val + b_val = " << sum << " qdlib" << endl;
    QUAD_ADD_IEEE(a_val, b_val, sum);
    cout << "a_val + b_val = " << sum << " macro" << endl;
    QUAD_ADD_SLOPPY(a_val, b_val, sum);
    cout << "a_val + b_val = " << sum << " sloppy" << endl;

    cout << endl;

    dd_real product = a_val * b_val;
    cout << "a_val * b_val = " << product << " qdlib" << endl;
    QUAD_MUL(a_val, b_val, product);
    cout << "a_val * b_val = " << product << " macro" << endl;
    QUAD_MUL_SLOPPY(a_val, b_val, product);
    cout << "a_val * b_val = " << product << " sloppy" << endl;

    return 0;
}
