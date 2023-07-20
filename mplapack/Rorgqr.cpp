/*
 * Copyright (c) 2008-2021
 *      Nakata, Maho
 *      All rights reserved.
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

#include <mpblas.h>
#include <mplapack.h>

void Rorgqr(INTEGER const m, INTEGER const n, INTEGER const k, REAL *a, INTEGER const lda, REAL *tau, REAL *work, INTEGER const lwork, INTEGER &info) {
    //
    //     Test the input arguments
    //
    info = 0;
    INTEGER nb = iMlaenv(1, "Rorgqr", " ", m, n, k, -1);
    INTEGER lwkopt = max((INTEGER)1, n) * nb;
    work[1 - 1] = lwkopt;
    bool lquery = (lwork == -1);
    if (m < 0) {
        info = -1;
    } else if (n < 0 || n > m) {
        info = -2;
    } else if (k < 0 || k > n) {
        info = -3;
    } else if (lda < max((INTEGER)1, m)) {
        info = -5;
    } else if (lwork < max((INTEGER)1, n) && !lquery) {
        info = -8;
    }
    if (info != 0) {
        Mxerbla("Rorgqr", -info);
        return;
    } else if (lquery) {
        return;
    }
    //
    //     Quick return if possible
    //
    if (n <= 0) {
        work[1 - 1] = 1;
        return;
    }
    //
    INTEGER nbmin = 2;
    INTEGER nx = 0;
    INTEGER iws = n;
    INTEGER ldwork = 0;
    if (nb > 1 && nb < k) {
        //
        //        Determine when to cross over from blocked to unblocked code.
        //
        nx = max((INTEGER)0, iMlaenv(3, "Rorgqr", " ", m, n, k, -1));
        if (nx < k) {
            //
            //           Determine if workspace is large enough for blocked code.
            //
            ldwork = n;
            iws = ldwork * nb;
            if (lwork < iws) {
                //
                //              Not enough workspace to use optimal NB:  reduce NB and
                //              determine the minimum value of NB.
                //
                nb = lwork / ldwork;
                nbmin = max((INTEGER)2, iMlaenv(2, "Rorgqr", " ", m, n, k, -1));
            }
        }
    }
    //
    INTEGER ki = 0;
    INTEGER kk = 0;
    INTEGER j = 0;
    INTEGER i = 0;
    const REAL zero = 0.0;
    if (nb >= nbmin && nb < k && nx < k) {
        //
        //        Use blocked code after the last block.
        //        The first kk columns are handled by the block method.
        //
        ki = ((k - nx - 1) / nb) * nb;
        kk = min(k, ki + nb);
        //
        //        Set A(1:kk,kk+1:n) to zero.
        //
        for (j = kk + 1; j <= n; j = j + 1) {
            for (i = 1; i <= kk; i = i + 1) {
                a[(i - 1) + (j - 1) * lda] = zero;
            }
        }
    } else {
        kk = 0;
    }
    //
    //     Use unblocked code for the last or only block.
    //
    INTEGER iinfo = 0;
    if (kk < n) {
        Rorg2r(m - kk, n - kk, k - kk, &a[((kk + 1) - 1) + ((kk + 1) - 1) * lda], lda, &tau[(kk + 1) - 1], work, iinfo);
    }
    //
    INTEGER ib = 0;
    INTEGER l = 0;
    if (kk > 0) {
        //
        //        Use blocked code
        //
        for (i = ki + 1; i >= 1; i = i - nb) {
            ib = min(nb, k - i + 1);
            if (i + ib <= n) {
                //
                //              Form the triangular factor of the block reflector
                //              H = H(i) H(i+1) . . . H(i+ib-1)
                //
                Rlarft("Forward", "Columnwise", m - i + 1, ib, &a[(i - 1) + (i - 1) * lda], lda, &tau[i - 1], work, ldwork);
                //
                //              Apply H to A(i:m,i+ib:n) from the left
                //
                Rlarfb("Left", "No transpose", "Forward", "Columnwise", m - i + 1, n - i - ib + 1, ib, &a[(i - 1) + (i - 1) * lda], lda, work, ldwork, &a[(i - 1) + ((i + ib) - 1) * lda], lda, &work[(ib + 1) - 1], ldwork);
            }
            //
            //           Apply H to rows i:m of current block
            //
            Rorg2r(m - i + 1, ib, ib, &a[(i - 1) + (i - 1) * lda], lda, &tau[i - 1], work, iinfo);
            //
            //           Set rows 1:i-1 of current block to zero
            //
            for (j = i; j <= i + ib - 1; j = j + 1) {
                for (l = 1; l <= i - 1; l = l + 1) {
                    a[(l - 1) + (j - 1) * lda] = zero;
                }
            }
        }
    }
    //
    work[1 - 1] = iws;
    //
    //     End of Rorgqr
    //
}
