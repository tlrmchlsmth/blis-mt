/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

#ifdef BLIS_ENABLE_BLAS2BLIS

#include "bli_f2c.h"

/* chpr2.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(c,hpr2)(char *uplo, integer *n, complex *alpha, complex *
	x, integer *incx, complex *y, integer *incy, complex *ap)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5, i__6;
    real r__1;
    complex q__1, q__2, q__3, q__4;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    integer info;
    complex temp1, temp2;
    integer i__, j, k;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    integer kk, ix, iy, jx = 0, jy = 0, kx = 0, ky = 0;
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CHPR2  performs the hermitian rank 2 operation */

/*     A := alpha*x*conjg( y' ) + conjg( alpha )*y*conjg( x' ) + A, */

/*  where alpha is a scalar, x and y are n element vectors and A is an */
/*  n by n hermitian matrix, supplied in packed form. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the upper or lower */
/*           triangular part of the matrix A is supplied in the packed */
/*           array AP as follows: */

/*              UPLO = 'U' or 'u'   The upper triangular part of A is */
/*                                  supplied in AP. */

/*              UPLO = 'L' or 'l'   The lower triangular part of A is */
/*                                  supplied in AP. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  ALPHA  - COMPLEX         . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  X      - COMPLEX          array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  Y      - COMPLEX          array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ). */
/*           Before entry, the incremented array Y must contain the n */
/*           element vector y. */
/*           Unchanged on exit. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */

/*  AP     - COMPLEX          array of DIMENSION at least */
/*           ( ( n*( n + 1 ) )/2 ). */
/*           Before entry with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular part of the hermitian matrix */
/*           packed sequentially, column by column, so that AP( 1 ) */
/*           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 1, 2 ) */
/*           and a( 2, 2 ) respectively, and so on. On exit, the array */
/*           AP is overwritten by the upper triangular part of the */
/*           updated matrix. */
/*           Before entry with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular part of the hermitian matrix */
/*           packed sequentially, column by column, so that AP( 1 ) */
/*           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 2, 1 ) */
/*           and a( 3, 1 ) respectively, and so on. On exit, the array */
/*           AP is overwritten by the lower triangular part of the */
/*           updated matrix. */
/*           Note that the imaginary parts of the diagonal elements need */
/*           not be set, they are assumed to be zero, and on exit they */
/*           are set to zero. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --ap;
    --y;
    --x;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U", (ftnlen)1, (ftnlen)1) && ! lsame_(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*incx == 0) {
	info = 5;
    } else if (*incy == 0) {
	info = 7;
    }
    if (info != 0) {
	xerbla_("CHPR2 ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || (alpha->r == 0.f && alpha->i == 0.f)) {
	return 0;
    }

/*     Set up the start points in X and Y if the increments are not both */
/*     unity. */

    if (*incx != 1 || *incy != 1) {
	if (*incx > 0) {
	    kx = 1;
	} else {
	    kx = 1 - (*n - 1) * *incx;
	}
	if (*incy > 0) {
	    ky = 1;
	} else {
	    ky = 1 - (*n - 1) * *incy;
	}
	jx = kx;
	jy = ky;
    }

/*     Start the operations. In this version the elements of the array AP */
/*     are accessed sequentially with one pass through AP. */

    kk = 1;
    if (lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {

/*        Form  A  when upper triangle is stored in AP. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		i__3 = j;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f || (y[i__3].r != 0.f 
			|| y[i__3].i != 0.f)) {
		    r_cnjg(&q__2, &y[j]);
		    q__1.r = alpha->r * q__2.r - alpha->i * q__2.i, q__1.i = 
			    alpha->r * q__2.i + alpha->i * q__2.r;
		    temp1.r = q__1.r, temp1.i = q__1.i;
		    i__2 = j;
		    q__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    r_cnjg(&q__1, &q__2);
		    temp2.r = q__1.r, temp2.i = q__1.i;
		    k = kk;
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			q__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i, 
				q__3.i = x[i__5].r * temp1.i + x[i__5].i * 
				temp1.r;
			q__2.r = ap[i__4].r + q__3.r, q__2.i = ap[i__4].i + 
				q__3.i;
			i__6 = i__;
			q__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i, 
				q__4.i = y[i__6].r * temp2.i + y[i__6].i * 
				temp2.r;
			q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
			ap[i__3].r = q__1.r, ap[i__3].i = q__1.i;
			++k;
/* L10: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = j;
		    q__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i, 
			    q__2.i = x[i__4].r * temp1.i + x[i__4].i * 
			    temp1.r;
		    i__5 = j;
		    q__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i, 
			    q__3.i = y[i__5].r * temp2.i + y[i__5].i * 
			    temp2.r;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    r__1 = ap[i__3].r + q__1.r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    r__1 = ap[i__3].r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		}
		kk += j;
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		i__3 = jy;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f || (y[i__3].r != 0.f 
			|| y[i__3].i != 0.f)) {
		    r_cnjg(&q__2, &y[jy]);
		    q__1.r = alpha->r * q__2.r - alpha->i * q__2.i, q__1.i = 
			    alpha->r * q__2.i + alpha->i * q__2.r;
		    temp1.r = q__1.r, temp1.i = q__1.i;
		    i__2 = jx;
		    q__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    r_cnjg(&q__1, &q__2);
		    temp2.r = q__1.r, temp2.i = q__1.i;
		    ix = kx;
		    iy = ky;
		    i__2 = kk + j - 2;
		    for (k = kk; k <= i__2; ++k) {
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			q__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i, 
				q__3.i = x[i__5].r * temp1.i + x[i__5].i * 
				temp1.r;
			q__2.r = ap[i__4].r + q__3.r, q__2.i = ap[i__4].i + 
				q__3.i;
			i__6 = iy;
			q__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i, 
				q__4.i = y[i__6].r * temp2.i + y[i__6].i * 
				temp2.r;
			q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
			ap[i__3].r = q__1.r, ap[i__3].i = q__1.i;
			ix += *incx;
			iy += *incy;
/* L30: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = jx;
		    q__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i, 
			    q__2.i = x[i__4].r * temp1.i + x[i__4].i * 
			    temp1.r;
		    i__5 = jy;
		    q__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i, 
			    q__3.i = y[i__5].r * temp2.i + y[i__5].i * 
			    temp2.r;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    r__1 = ap[i__3].r + q__1.r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    r__1 = ap[i__3].r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		}
		jx += *incx;
		jy += *incy;
		kk += j;
/* L40: */
	    }
	}
    } else {

/*        Form  A  when lower triangle is stored in AP. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		i__3 = j;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f || (y[i__3].r != 0.f 
			|| y[i__3].i != 0.f)) {
		    r_cnjg(&q__2, &y[j]);
		    q__1.r = alpha->r * q__2.r - alpha->i * q__2.i, q__1.i = 
			    alpha->r * q__2.i + alpha->i * q__2.r;
		    temp1.r = q__1.r, temp1.i = q__1.i;
		    i__2 = j;
		    q__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    r_cnjg(&q__1, &q__2);
		    temp2.r = q__1.r, temp2.i = q__1.i;
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = j;
		    q__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i, 
			    q__2.i = x[i__4].r * temp1.i + x[i__4].i * 
			    temp1.r;
		    i__5 = j;
		    q__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i, 
			    q__3.i = y[i__5].r * temp2.i + y[i__5].i * 
			    temp2.r;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    r__1 = ap[i__3].r + q__1.r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		    k = kk + 1;
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			q__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i, 
				q__3.i = x[i__5].r * temp1.i + x[i__5].i * 
				temp1.r;
			q__2.r = ap[i__4].r + q__3.r, q__2.i = ap[i__4].i + 
				q__3.i;
			i__6 = i__;
			q__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i, 
				q__4.i = y[i__6].r * temp2.i + y[i__6].i * 
				temp2.r;
			q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
			ap[i__3].r = q__1.r, ap[i__3].i = q__1.i;
			++k;
/* L50: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    r__1 = ap[i__3].r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		}
		kk = kk + *n - j + 1;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		i__3 = jy;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f || (y[i__3].r != 0.f 
			|| y[i__3].i != 0.f)) {
		    r_cnjg(&q__2, &y[jy]);
		    q__1.r = alpha->r * q__2.r - alpha->i * q__2.i, q__1.i = 
			    alpha->r * q__2.i + alpha->i * q__2.r;
		    temp1.r = q__1.r, temp1.i = q__1.i;
		    i__2 = jx;
		    q__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    r_cnjg(&q__1, &q__2);
		    temp2.r = q__1.r, temp2.i = q__1.i;
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = jx;
		    q__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i, 
			    q__2.i = x[i__4].r * temp1.i + x[i__4].i * 
			    temp1.r;
		    i__5 = jy;
		    q__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i, 
			    q__3.i = y[i__5].r * temp2.i + y[i__5].i * 
			    temp2.r;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    r__1 = ap[i__3].r + q__1.r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		    ix = jx;
		    iy = jy;
		    i__2 = kk + *n - j;
		    for (k = kk + 1; k <= i__2; ++k) {
			ix += *incx;
			iy += *incy;
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			q__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i, 
				q__3.i = x[i__5].r * temp1.i + x[i__5].i * 
				temp1.r;
			q__2.r = ap[i__4].r + q__3.r, q__2.i = ap[i__4].i + 
				q__3.i;
			i__6 = iy;
			q__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i, 
				q__4.i = y[i__6].r * temp2.i + y[i__6].i * 
				temp2.r;
			q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
			ap[i__3].r = q__1.r, ap[i__3].i = q__1.i;
/* L70: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    r__1 = ap[i__3].r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		}
		jx += *incx;
		jy += *incy;
		kk = kk + *n - j + 1;
/* L80: */
	    }
	}
    }

    return 0;

/*     End of CHPR2 . */

} /* chpr2_ */

/* zhpr2.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(z,hpr2)(char *uplo, integer *n, doublecomplex *alpha, 
	doublecomplex *x, integer *incx, doublecomplex *y, integer *incy, 
	doublecomplex *ap)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5, i__6;
    doublereal d__1;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer info;
    doublecomplex temp1, temp2;
    integer i__, j, k;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    integer kk, ix, iy, jx = 0, jy = 0, kx = 0, ky = 0;
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZHPR2  performs the hermitian rank 2 operation */

/*     A := alpha*x*conjg( y' ) + conjg( alpha )*y*conjg( x' ) + A, */

/*  where alpha is a scalar, x and y are n element vectors and A is an */
/*  n by n hermitian matrix, supplied in packed form. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the upper or lower */
/*           triangular part of the matrix A is supplied in the packed */
/*           array AP as follows: */

/*              UPLO = 'U' or 'u'   The upper triangular part of A is */
/*                                  supplied in AP. */

/*              UPLO = 'L' or 'l'   The lower triangular part of A is */
/*                                  supplied in AP. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  ALPHA  - COMPLEX*16      . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  X      - COMPLEX*16       array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  Y      - COMPLEX*16       array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ). */
/*           Before entry, the incremented array Y must contain the n */
/*           element vector y. */
/*           Unchanged on exit. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */

/*  AP     - COMPLEX*16       array of DIMENSION at least */
/*           ( ( n*( n + 1 ) )/2 ). */
/*           Before entry with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular part of the hermitian matrix */
/*           packed sequentially, column by column, so that AP( 1 ) */
/*           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 1, 2 ) */
/*           and a( 2, 2 ) respectively, and so on. On exit, the array */
/*           AP is overwritten by the upper triangular part of the */
/*           updated matrix. */
/*           Before entry with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular part of the hermitian matrix */
/*           packed sequentially, column by column, so that AP( 1 ) */
/*           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 2, 1 ) */
/*           and a( 3, 1 ) respectively, and so on. On exit, the array */
/*           AP is overwritten by the lower triangular part of the */
/*           updated matrix. */
/*           Note that the imaginary parts of the diagonal elements need */
/*           not be set, they are assumed to be zero, and on exit they */
/*           are set to zero. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --ap;
    --y;
    --x;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U", (ftnlen)1, (ftnlen)1) && ! lsame_(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*incx == 0) {
	info = 5;
    } else if (*incy == 0) {
	info = 7;
    }
    if (info != 0) {
	xerbla_("ZHPR2 ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || (alpha->r == 0. && alpha->i == 0.)) {
	return 0;
    }

/*     Set up the start points in X and Y if the increments are not both */
/*     unity. */

    if (*incx != 1 || *incy != 1) {
	if (*incx > 0) {
	    kx = 1;
	} else {
	    kx = 1 - (*n - 1) * *incx;
	}
	if (*incy > 0) {
	    ky = 1;
	} else {
	    ky = 1 - (*n - 1) * *incy;
	}
	jx = kx;
	jy = ky;
    }

/*     Start the operations. In this version the elements of the array AP */
/*     are accessed sequentially with one pass through AP. */

    kk = 1;
    if (lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {

/*        Form  A  when upper triangle is stored in AP. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		i__3 = j;
		if (x[i__2].r != 0. || x[i__2].i != 0. || (y[i__3].r != 0. || 
			y[i__3].i != 0.)) {
		    d_cnjg(&z__2, &y[j]);
		    z__1.r = alpha->r * z__2.r - alpha->i * z__2.i, z__1.i = 
			    alpha->r * z__2.i + alpha->i * z__2.r;
		    temp1.r = z__1.r, temp1.i = z__1.i;
		    i__2 = j;
		    z__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    z__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    d_cnjg(&z__1, &z__2);
		    temp2.r = z__1.r, temp2.i = z__1.i;
		    k = kk;
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			z__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i, 
				z__3.i = x[i__5].r * temp1.i + x[i__5].i * 
				temp1.r;
			z__2.r = ap[i__4].r + z__3.r, z__2.i = ap[i__4].i + 
				z__3.i;
			i__6 = i__;
			z__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i, 
				z__4.i = y[i__6].r * temp2.i + y[i__6].i * 
				temp2.r;
			z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
			ap[i__3].r = z__1.r, ap[i__3].i = z__1.i;
			++k;
/* L10: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = j;
		    z__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i, 
			    z__2.i = x[i__4].r * temp1.i + x[i__4].i * 
			    temp1.r;
		    i__5 = j;
		    z__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i, 
			    z__3.i = y[i__5].r * temp2.i + y[i__5].i * 
			    temp2.r;
		    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
		    d__1 = ap[i__3].r + z__1.r;
		    ap[i__2].r = d__1, ap[i__2].i = 0.;
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    d__1 = ap[i__3].r;
		    ap[i__2].r = d__1, ap[i__2].i = 0.;
		}
		kk += j;
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		i__3 = jy;
		if (x[i__2].r != 0. || x[i__2].i != 0. || (y[i__3].r != 0. || 
			y[i__3].i != 0.)) {
		    d_cnjg(&z__2, &y[jy]);
		    z__1.r = alpha->r * z__2.r - alpha->i * z__2.i, z__1.i = 
			    alpha->r * z__2.i + alpha->i * z__2.r;
		    temp1.r = z__1.r, temp1.i = z__1.i;
		    i__2 = jx;
		    z__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    z__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    d_cnjg(&z__1, &z__2);
		    temp2.r = z__1.r, temp2.i = z__1.i;
		    ix = kx;
		    iy = ky;
		    i__2 = kk + j - 2;
		    for (k = kk; k <= i__2; ++k) {
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			z__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i, 
				z__3.i = x[i__5].r * temp1.i + x[i__5].i * 
				temp1.r;
			z__2.r = ap[i__4].r + z__3.r, z__2.i = ap[i__4].i + 
				z__3.i;
			i__6 = iy;
			z__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i, 
				z__4.i = y[i__6].r * temp2.i + y[i__6].i * 
				temp2.r;
			z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
			ap[i__3].r = z__1.r, ap[i__3].i = z__1.i;
			ix += *incx;
			iy += *incy;
/* L30: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = jx;
		    z__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i, 
			    z__2.i = x[i__4].r * temp1.i + x[i__4].i * 
			    temp1.r;
		    i__5 = jy;
		    z__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i, 
			    z__3.i = y[i__5].r * temp2.i + y[i__5].i * 
			    temp2.r;
		    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
		    d__1 = ap[i__3].r + z__1.r;
		    ap[i__2].r = d__1, ap[i__2].i = 0.;
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    d__1 = ap[i__3].r;
		    ap[i__2].r = d__1, ap[i__2].i = 0.;
		}
		jx += *incx;
		jy += *incy;
		kk += j;
/* L40: */
	    }
	}
    } else {

/*        Form  A  when lower triangle is stored in AP. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		i__3 = j;
		if (x[i__2].r != 0. || x[i__2].i != 0. || (y[i__3].r != 0. || 
			y[i__3].i != 0.)) {
		    d_cnjg(&z__2, &y[j]);
		    z__1.r = alpha->r * z__2.r - alpha->i * z__2.i, z__1.i = 
			    alpha->r * z__2.i + alpha->i * z__2.r;
		    temp1.r = z__1.r, temp1.i = z__1.i;
		    i__2 = j;
		    z__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    z__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    d_cnjg(&z__1, &z__2);
		    temp2.r = z__1.r, temp2.i = z__1.i;
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = j;
		    z__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i, 
			    z__2.i = x[i__4].r * temp1.i + x[i__4].i * 
			    temp1.r;
		    i__5 = j;
		    z__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i, 
			    z__3.i = y[i__5].r * temp2.i + y[i__5].i * 
			    temp2.r;
		    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
		    d__1 = ap[i__3].r + z__1.r;
		    ap[i__2].r = d__1, ap[i__2].i = 0.;
		    k = kk + 1;
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			z__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i, 
				z__3.i = x[i__5].r * temp1.i + x[i__5].i * 
				temp1.r;
			z__2.r = ap[i__4].r + z__3.r, z__2.i = ap[i__4].i + 
				z__3.i;
			i__6 = i__;
			z__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i, 
				z__4.i = y[i__6].r * temp2.i + y[i__6].i * 
				temp2.r;
			z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
			ap[i__3].r = z__1.r, ap[i__3].i = z__1.i;
			++k;
/* L50: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    d__1 = ap[i__3].r;
		    ap[i__2].r = d__1, ap[i__2].i = 0.;
		}
		kk = kk + *n - j + 1;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		i__3 = jy;
		if (x[i__2].r != 0. || x[i__2].i != 0. || (y[i__3].r != 0. || 
			y[i__3].i != 0.)) {
		    d_cnjg(&z__2, &y[jy]);
		    z__1.r = alpha->r * z__2.r - alpha->i * z__2.i, z__1.i = 
			    alpha->r * z__2.i + alpha->i * z__2.r;
		    temp1.r = z__1.r, temp1.i = z__1.i;
		    i__2 = jx;
		    z__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    z__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    d_cnjg(&z__1, &z__2);
		    temp2.r = z__1.r, temp2.i = z__1.i;
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = jx;
		    z__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i, 
			    z__2.i = x[i__4].r * temp1.i + x[i__4].i * 
			    temp1.r;
		    i__5 = jy;
		    z__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i, 
			    z__3.i = y[i__5].r * temp2.i + y[i__5].i * 
			    temp2.r;
		    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
		    d__1 = ap[i__3].r + z__1.r;
		    ap[i__2].r = d__1, ap[i__2].i = 0.;
		    ix = jx;
		    iy = jy;
		    i__2 = kk + *n - j;
		    for (k = kk + 1; k <= i__2; ++k) {
			ix += *incx;
			iy += *incy;
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			z__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i, 
				z__3.i = x[i__5].r * temp1.i + x[i__5].i * 
				temp1.r;
			z__2.r = ap[i__4].r + z__3.r, z__2.i = ap[i__4].i + 
				z__3.i;
			i__6 = iy;
			z__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i, 
				z__4.i = y[i__6].r * temp2.i + y[i__6].i * 
				temp2.r;
			z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
			ap[i__3].r = z__1.r, ap[i__3].i = z__1.i;
/* L70: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    d__1 = ap[i__3].r;
		    ap[i__2].r = d__1, ap[i__2].i = 0.;
		}
		jx += *incx;
		jy += *incy;
		kk = kk + *n - j + 1;
/* L80: */
	    }
	}
    }

    return 0;

/*     End of ZHPR2 . */

} /* zhpr2_ */

#endif

