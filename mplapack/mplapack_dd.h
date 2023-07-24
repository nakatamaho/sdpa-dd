/*
 * Copyright (c) 2008-2021
 *	Nakata, Maho
 * 	All rights reserved.
 *
 * $Id: mplapack_dd.h,v 1.31 2010/08/07 03:15:46 nakatamaho Exp $
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

#ifndef _MPLAPACK_DD_H_
#define _MPLAPACK_DD_H_

/* this is a subset of mplapack only for SDPA-DD */
#include "mplapack_config.h"
#include "qd/dd_real.h"

dd_real Rlamch_dd(const char *cmach);
void Rsyev(const char *jobz, const char *uplo, mplapackint const n, dd_real *a, mplapackint const lda, dd_real *w, dd_real *work, mplapackint const lwork, mplapackint &info);
void Rsteqr(const char *compz, mplapackint const n, dd_real *d, dd_real *e, dd_real *z, mplapackint const ldz, dd_real *work, mplapackint &info);
void Rpotrf(const char *uplo, mplapackint const n, dd_real *a, mplapackint const lda, mplapackint &info);
void Rpotrf2(const char *uplo, mplapackint const n, dd_real *a, mplapackint const lda, mplapackint &info);
void Rlascl(const char *type, mplapackint const kl, mplapackint const ku, dd_real const cfrom, dd_real const cto, mplapackint const m, mplapackint const n, dd_real *a, mplapackint const lda, mplapackint &info);
void Rlascl2(mplapackint const m, mplapackint const n, dd_real *d, dd_real *x, mplapackint const ldx);
void Rsytrd(const char *uplo, mplapackint const n, dd_real *a, mplapackint const lda, dd_real *d, dd_real *e, dd_real *tau, dd_real *work, mplapackint const lwork, mplapackint &info);
void Rsytrd_2stage(const char *vect, const char *uplo, mplapackint const n, dd_real *a, mplapackint const lda, dd_real *d, dd_real *e, dd_real *tau, dd_real *hous2, mplapackint const lhous2, dd_real *work, mplapackint const lwork, mplapackint &info);
void Rsytrd_sb2st(const char *stage1, const char *vect, const char *uplo, mplapackint const n, mplapackint const kd, dd_real *ab, mplapackint const ldab, dd_real *d, dd_real *e, dd_real *hous, mplapackint const lhous, dd_real *work, mplapackint const lwork, mplapackint &info);
void Rsytrd_sy2sb(const char *uplo, mplapackint const n, mplapackint const kd, dd_real *a, mplapackint const lda, dd_real *ab, mplapackint const ldab, dd_real *tau, dd_real *work, mplapackint const lwork, mplapackint &info);
void Rsytd2(const char *uplo, mplapackint const n, dd_real *a, mplapackint const lda, dd_real *d, dd_real *e, dd_real *tau, mplapackint &info);
void Rlae2(dd_real const a, dd_real const b, dd_real const c, dd_real &rt1, dd_real &rt2);
void Rlasrt(const char *id, mplapackint const n, dd_real *d, mplapackint &info);
void Rorgql(mplapackint const m, mplapackint const n, mplapackint const k, dd_real *a, mplapackint const lda, dd_real *tau, dd_real *work, mplapackint const lwork, mplapackint &info);
void Rorgqr(mplapackint const m, mplapackint const n, mplapackint const k, dd_real *a, mplapackint const lda, dd_real *tau, dd_real *work, mplapackint const lwork, mplapackint &info);
void Rlarfg(mplapackint const n, dd_real &alpha, dd_real *x, mplapackint const incx, dd_real &tau);
void Rlarfgp(mplapackint const n, dd_real &alpha, dd_real *x, mplapackint const incx, dd_real &tau);
void Rlassq(mplapackint const n, dd_real *x, mplapackint const incx, dd_real &scale, dd_real &sumsq);
void Rorg2l(mplapackint const m, mplapackint const n, mplapackint const k, dd_real *a, mplapackint const lda, dd_real *tau, dd_real *work, mplapackint &info);
void Rlarft(const char *direct, const char *storev, mplapackint const n, mplapackint const k, dd_real *v, mplapackint const ldv, dd_real *tau, dd_real *t, mplapackint const ldt);
void Rlarfb(const char *side, const char *trans, const char *direct, const char *storev, mplapackint const m, mplapackint const n, mplapackint const k, dd_real *v, mplapackint const ldv, dd_real *t, mplapackint const ldt, dd_real *c, mplapackint const ldc, dd_real *work, mplapackint const ldwork);
void Rlarfb_gett(const char *ident, mplapackint const m, mplapackint const n, mplapackint const k, dd_real *t, mplapackint const ldt, dd_real *a, mplapackint const lda, dd_real *b, mplapackint const ldb, dd_real *work, mplapackint const ldwork);
void Rorg2r(mplapackint const m, mplapackint const n, mplapackint const k, dd_real *a, mplapackint const lda, dd_real *tau, dd_real *work, mplapackint &info);
void Rlarf(const char *side, mplapackint const m, mplapackint const n, dd_real *v, mplapackint const incv, dd_real const tau, dd_real *c, mplapackint const ldc, dd_real *work);
void Rlarfb(const char *side, const char *trans, const char *direct, const char *storev, mplapackint const m, mplapackint const n, mplapackint const k, dd_real *v, mplapackint const ldv, dd_real *t, mplapackint const ldt, dd_real *c, mplapackint const ldc, dd_real *work, mplapackint const ldwork);
void Rlarfb_gett(const char *ident, mplapackint const m, mplapackint const n, mplapackint const k, dd_real *t, mplapackint const ldt, dd_real *a, mplapackint const lda, dd_real *b, mplapackint const ldb, dd_real *work, mplapackint const ldwork);
void Rlarfg(mplapackint const n, dd_real &alpha, dd_real *x, mplapackint const incx, dd_real &tau);
void Rlarfgp(mplapackint const n, dd_real &alpha, dd_real *x, mplapackint const incx, dd_real &tau);
void Rlarft(const char *direct, const char *storev, mplapackint const n, mplapackint const k, dd_real *v, mplapackint const ldv, dd_real *tau, dd_real *t, mplapackint const ldt);
void Rlarfx(const char *side, mplapackint const m, mplapackint const n, dd_real *v, dd_real const tau, dd_real *c, mplapackint const ldc, dd_real *work);
void Rlarfy(const char *uplo, mplapackint const n, dd_real *v, mplapackint const incv, dd_real const tau, dd_real *c, mplapackint const ldc, dd_real *work);
void Rpotf2(const char *uplo, mplapackint const n, dd_real *a, mplapackint const lda, mplapackint &info);
void Rlaset(const char *uplo, mplapackint const m, mplapackint const n, dd_real const alpha, dd_real const beta, dd_real *a, mplapackint const lda);
void Rlaev2(dd_real const a, dd_real const b, dd_real const c, dd_real &rt1, dd_real &rt2, dd_real &cs1, dd_real &sn1);
void Rlasr(const char *side, const char *pivot, const char *direct, mplapackint const m, mplapackint const n, dd_real *c, dd_real *s, dd_real *a, mplapackint const lda);
void Rlasrt(const char *id, mplapackint const n, dd_real *d, mplapackint &info);
void Rlartg(dd_real const f, dd_real const g, dd_real &cs, dd_real &sn, dd_real &r);
void Rlartgp(dd_real const f, dd_real const g, dd_real &cs, dd_real &sn, dd_real &r);
void Rlartgs(dd_real const x, dd_real const y, dd_real const sigma, dd_real &cs, dd_real &sn);
void Rlatrd(const char *uplo, mplapackint const n, mplapackint const nb, dd_real *a, mplapackint const lda, dd_real *e, dd_real *tau, dd_real *w, mplapackint const ldw);
void Rsterf(mplapackint const n, dd_real *d, dd_real *e, mplapackint &info);
void Rorgtr(const char *uplo, mplapackint const n, dd_real *a, mplapackint const lda, dd_real *tau, dd_real *work, mplapackint const lwork, mplapackint &info);
mplapackint iMparmq_dd(mplapackint const ispec, const char *name, const char *opts, mplapackint const n, mplapackint const ilo, mplapackint const ihi, mplapackint const lwork);
mplapackint iMieeeck_dd(mplapackint const &ispec, dd_real const &zero, dd_real const &one);
bool Rlaisnan(dd_real const din1, dd_real const din2);
bool Risnan(dd_real const din);
mplapackint iMlaenv_dd(mplapackint ispec, const char *name, const char *opts, mplapackint n1, mplapackint n2, mplapackint n3, mplapackint n4);
dd_real Rlanst(const char *norm, mplapackint const n, dd_real *d, dd_real *e);
dd_real Rlapy2(dd_real const x, dd_real const y);
mplapackint iMladlr(mplapackint const m, mplapackint const n, dd_real *a, mplapackint const lda);
mplapackint iMladlc(mplapackint const m, mplapackint const n, dd_real *a, mplapackint const lda);
dd_real Rlansy(const char *norm, const char *uplo, mplapackint const n, dd_real *a, mplapackint const lda, dd_real *work);
void Rcombssq(dd_real *v1, dd_real *v2);

#endif
