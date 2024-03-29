/*
 * Copyright (c) 2008-2021
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

#ifndef _MUTILS_DD_H_
#define _MUTILS_DD_H_

inline dd_real pow2(dd_real a) {
    dd_real mtmp = a * a;
    return mtmp;
}

// implementation of sign transfer function.
inline dd_real sign(dd_real a, dd_real b) {
    dd_real mtmp;
    mtmp = abs(a);
    if (b < 0.0) {
        mtmp = -mtmp;
    }
    return mtmp;
}

inline dd_real castREAL_dd(mplapackint n) {
    dd_real ret;
    ret.x[0] = (static_cast<double>(n));
    ret.x[1] = 0.0;
    return ret;
}
inline mplapackint castINTEGER_dd(dd_real a) {
    mplapackint i = a.x[0];
    return i;
}

inline long __dd_nint(dd_real a) {
    long i;
    dd_real tmp;
    a = a + 0.5;
    tmp = floor(a);
    i = (int)tmp.x[0];
    return i;
}

inline double cast2double(dd_real a) { return a.x[0]; }

#endif
