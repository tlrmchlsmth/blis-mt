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

/* cgbmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(c,gbmv)(char *trans, integer *m, integer *n, integer *kl, 
	integer *ku, complex *alpha, complex *a, integer *lda, complex *x, 
	integer *incx, complex *beta, complex *y, integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    complex q__1, q__2, q__3;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    integer info;
    complex temp;
    integer lenx, leny, i__, j, k;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    integer ix, iy, jx, jy, kx, ky;
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);
    logical noconj;
    integer kup1;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CGBMV  performs one of the matrix-vector operations */

/*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,   or */

/*     y := alpha*conjg( A' )*x + beta*y, */

/*  where alpha and beta are scalars, x and y are vectors and A is an */
/*  m by n band matrix, with kl sub-diagonals and ku super-diagonals. */

/*  Parameters */
/*  ========== */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y. */

/*              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y. */

/*              TRANS = 'C' or 'c'   y := alpha*conjg( A' )*x + beta*y. */

/*           Unchanged on exit. */

/*  M      - INTEGER. */
/*           On entry, M specifies the number of rows of the matrix A. */
/*           M must be at least zero. */
/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the number of columns of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  KL     - INTEGER. */
/*           On entry, KL specifies the number of sub-diagonals of the */
/*           matrix A. KL must satisfy  0 .le. KL. */
/*           Unchanged on exit. */

/*  KU     - INTEGER. */
/*           On entry, KU specifies the number of super-diagonals of the */
/*           matrix A. KU must satisfy  0 .le. KU. */
/*           Unchanged on exit. */

/*  ALPHA  - COMPLEX         . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - COMPLEX          array of DIMENSION ( LDA, n ). */
/*           Before entry, the leading ( kl + ku + 1 ) by n part of the */
/*           array A must contain the matrix of coefficients, supplied */
/*           column by column, with the leading diagonal of the matrix in */
/*           row ( ku + 1 ) of the array, the first super-diagonal */
/*           starting at position 2 in row ku, the first sub-diagonal */
/*           starting at position 1 in row ( ku + 2 ), and so on. */
/*           Elements in the array A that do not correspond to elements */
/*           in the band matrix (such as the top left ku by ku triangle) */
/*           are not referenced. */
/*           The following program segment will transfer a band matrix */
/*           from conventional full matrix storage to band storage: */

/*                 DO 20, J = 1, N */
/*                    K = KU + 1 - J */
/*                    DO 10, I = MAX( 1, J - KU ), MIN( M, J + KL ) */
/*                       A( K + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( kl + ku + 1 ). */
/*           Unchanged on exit. */

/*  X      - COMPLEX          array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise. */
/*           Before entry, the incremented array X must contain the */
/*           vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  BETA   - COMPLEX         . */
/*           On entry, BETA specifies the scalar beta. When BETA is */
/*           supplied as zero then Y need not be set on input. */
/*           Unchanged on exit. */

/*  Y      - COMPLEX          array of DIMENSION at least */
/*           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise. */
/*           Before entry, the incremented array Y must contain the */
/*           vector y. On exit, Y is overwritten by the updated vector y. */


/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */


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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! lsame_(trans, "N", (ftnlen)1, (ftnlen)1) && ! lsame_(trans, "T", (
	    ftnlen)1, (ftnlen)1) && ! lsame_(trans, "C", (ftnlen)1, (ftnlen)1)
	    ) {
	info = 1;
    } else if (*m < 0) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*kl < 0) {
	info = 4;
    } else if (*ku < 0) {
	info = 5;
    } else if (*lda < *kl + *ku + 1) {
	info = 8;
    } else if (*incx == 0) {
	info = 10;
    } else if (*incy == 0) {
	info = 13;
    }
    if (info != 0) {
	xerbla_("CGBMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || (alpha->r == 0.f && alpha->i == 0.f && (beta->r 
	    == 1.f && beta->i == 0.f))) {
	return 0;
    }

    noconj = lsame_(trans, "T", (ftnlen)1, (ftnlen)1);

/*     Set  LENX  and  LENY, the lengths of the vectors x and y, and set */
/*     up the start points in  X  and  Y. */

    if (lsame_(trans, "N", (ftnlen)1, (ftnlen)1)) {
	lenx = *n;
	leny = *m;
    } else {
	lenx = *m;
	leny = *n;
    }
    if (*incx > 0) {
	kx = 1;
    } else {
	kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
	ky = 1;
    } else {
	ky = 1 - (leny - 1) * *incy;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through the band part of A. */

/*     First form  y := beta*y. */

    if (beta->r != 1.f || beta->i != 0.f) {
	if (*incy == 1) {
	    if (beta->r == 0.f && beta->i == 0.f) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    y[i__2].r = 0.f, y[i__2].i = 0.f;
/* L10: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    i__3 = i__;
		    q__1.r = beta->r * y[i__3].r - beta->i * y[i__3].i, 
			    q__1.i = beta->r * y[i__3].i + beta->i * y[i__3]
			    .r;
		    y[i__2].r = q__1.r, y[i__2].i = q__1.i;
/* L20: */
		}
	    }
	} else {
	    iy = ky;
	    if (beta->r == 0.f && beta->i == 0.f) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = iy;
		    y[i__2].r = 0.f, y[i__2].i = 0.f;
		    iy += *incy;
/* L30: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = iy;
		    i__3 = iy;
		    q__1.r = beta->r * y[i__3].r - beta->i * y[i__3].i, 
			    q__1.i = beta->r * y[i__3].i + beta->i * y[i__3]
			    .r;
		    y[i__2].r = q__1.r, y[i__2].i = q__1.i;
		    iy += *incy;
/* L40: */
		}
	    }
	}
    }
    if (alpha->r == 0.f && alpha->i == 0.f) {
	return 0;
    }
    kup1 = *ku + 1;
    if (lsame_(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  y := alpha*A*x + y. */

	jx = kx;
	if (*incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
		    i__2 = jx;
		    q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__1.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    temp.r = q__1.r, temp.i = q__1.i;
		    k = kup1 - j;
/* Computing MAX */
		    i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__4 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			i__2 = i__;
			i__3 = i__;
			i__5 = k + i__ + j * a_dim1;
			q__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i, 
				q__2.i = temp.r * a[i__5].i + temp.i * a[i__5]
				.r;
			q__1.r = y[i__3].r + q__2.r, q__1.i = y[i__3].i + 
				q__2.i;
			y[i__2].r = q__1.r, y[i__2].i = q__1.i;
/* L50: */
		    }
		}
		jx += *incx;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__4 = jx;
		if (x[i__4].r != 0.f || x[i__4].i != 0.f) {
		    i__4 = jx;
		    q__1.r = alpha->r * x[i__4].r - alpha->i * x[i__4].i, 
			    q__1.i = alpha->r * x[i__4].i + alpha->i * x[i__4]
			    .r;
		    temp.r = q__1.r, temp.i = q__1.i;
		    iy = ky;
		    k = kup1 - j;
/* Computing MAX */
		    i__4 = 1, i__2 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__3 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			i__4 = iy;
			i__2 = iy;
			i__5 = k + i__ + j * a_dim1;
			q__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i, 
				q__2.i = temp.r * a[i__5].i + temp.i * a[i__5]
				.r;
			q__1.r = y[i__2].r + q__2.r, q__1.i = y[i__2].i + 
				q__2.i;
			y[i__4].r = q__1.r, y[i__4].i = q__1.i;
			iy += *incy;
/* L70: */
		    }
		}
		jx += *incx;
		if (j > *ku) {
		    ky += *incy;
		}
/* L80: */
	    }
	}
    } else {

/*        Form  y := alpha*A'*x + y  or  y := alpha*conjg( A' )*x + y. */

	jy = ky;
	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp.r = 0.f, temp.i = 0.f;
		k = kup1 - j;
		if (noconj) {
/* Computing MAX */
		    i__3 = 1, i__4 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__2 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__3,i__4); i__ <= i__2; ++i__) {
			i__3 = k + i__ + j * a_dim1;
			i__4 = i__;
			q__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[i__4]
				.i, q__2.i = a[i__3].r * x[i__4].i + a[i__3]
				.i * x[i__4].r;
			q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
			temp.r = q__1.r, temp.i = q__1.i;
/* L90: */
		    }
		} else {
/* Computing MAX */
		    i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__4 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			r_cnjg(&q__3, &a[k + i__ + j * a_dim1]);
			i__2 = i__;
			q__2.r = q__3.r * x[i__2].r - q__3.i * x[i__2].i, 
				q__2.i = q__3.r * x[i__2].i + q__3.i * x[i__2]
				.r;
			q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
			temp.r = q__1.r, temp.i = q__1.i;
/* L100: */
		    }
		}
		i__4 = jy;
		i__2 = jy;
		q__2.r = alpha->r * temp.r - alpha->i * temp.i, q__2.i = 
			alpha->r * temp.i + alpha->i * temp.r;
		q__1.r = y[i__2].r + q__2.r, q__1.i = y[i__2].i + q__2.i;
		y[i__4].r = q__1.r, y[i__4].i = q__1.i;
		jy += *incy;
/* L110: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp.r = 0.f, temp.i = 0.f;
		ix = kx;
		k = kup1 - j;
		if (noconj) {
/* Computing MAX */
		    i__4 = 1, i__2 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__3 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			i__4 = k + i__ + j * a_dim1;
			i__2 = ix;
			q__2.r = a[i__4].r * x[i__2].r - a[i__4].i * x[i__2]
				.i, q__2.i = a[i__4].r * x[i__2].i + a[i__4]
				.i * x[i__2].r;
			q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
			temp.r = q__1.r, temp.i = q__1.i;
			ix += *incx;
/* L120: */
		    }
		} else {
/* Computing MAX */
		    i__3 = 1, i__4 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__2 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__3,i__4); i__ <= i__2; ++i__) {
			r_cnjg(&q__3, &a[k + i__ + j * a_dim1]);
			i__3 = ix;
			q__2.r = q__3.r * x[i__3].r - q__3.i * x[i__3].i, 
				q__2.i = q__3.r * x[i__3].i + q__3.i * x[i__3]
				.r;
			q__1.r = temp.r + q__2.r, q__1.i = temp.i + q__2.i;
			temp.r = q__1.r, temp.i = q__1.i;
			ix += *incx;
/* L130: */
		    }
		}
		i__2 = jy;
		i__3 = jy;
		q__2.r = alpha->r * temp.r - alpha->i * temp.i, q__2.i = 
			alpha->r * temp.i + alpha->i * temp.r;
		q__1.r = y[i__3].r + q__2.r, q__1.i = y[i__3].i + q__2.i;
		y[i__2].r = q__1.r, y[i__2].i = q__1.i;
		jy += *incy;
		if (j > *ku) {
		    kx += *incx;
		}
/* L140: */
	    }
	}
    }

    return 0;

/*     End of CGBMV . */

} /* cgbmv_ */

/* dgbmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(d,gbmv)(char *trans, integer *m, integer *n, integer *kl, 
	integer *ku, doublereal *alpha, doublereal *a, integer *lda, 
	doublereal *x, integer *incx, doublereal *beta, doublereal *y, 
	integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;

    /* Local variables */
    integer info;
    doublereal temp;
    integer lenx, leny, i__, j, k;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    integer ix, iy, jx, jy, kx, ky;
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);
    integer kup1;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DGBMV  performs one of the matrix-vector operations */

/*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y, */

/*  where alpha and beta are scalars, x and y are vectors and A is an */
/*  m by n band matrix, with kl sub-diagonals and ku super-diagonals. */

/*  Parameters */
/*  ========== */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y. */

/*              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y. */

/*              TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y. */

/*           Unchanged on exit. */

/*  M      - INTEGER. */
/*           On entry, M specifies the number of rows of the matrix A. */
/*           M must be at least zero. */
/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the number of columns of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  KL     - INTEGER. */
/*           On entry, KL specifies the number of sub-diagonals of the */
/*           matrix A. KL must satisfy  0 .le. KL. */
/*           Unchanged on exit. */

/*  KU     - INTEGER. */
/*           On entry, KU specifies the number of super-diagonals of the */
/*           matrix A. KU must satisfy  0 .le. KU. */
/*           Unchanged on exit. */

/*  ALPHA  - DOUBLE PRECISION. */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ). */
/*           Before entry, the leading ( kl + ku + 1 ) by n part of the */
/*           array A must contain the matrix of coefficients, supplied */
/*           column by column, with the leading diagonal of the matrix in */
/*           row ( ku + 1 ) of the array, the first super-diagonal */
/*           starting at position 2 in row ku, the first sub-diagonal */
/*           starting at position 1 in row ( ku + 2 ), and so on. */
/*           Elements in the array A that do not correspond to elements */
/*           in the band matrix (such as the top left ku by ku triangle) */
/*           are not referenced. */
/*           The following program segment will transfer a band matrix */
/*           from conventional full matrix storage to band storage: */

/*                 DO 20, J = 1, N */
/*                    K = KU + 1 - J */
/*                    DO 10, I = MAX( 1, J - KU ), MIN( M, J + KL ) */
/*                       A( K + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( kl + ku + 1 ). */
/*           Unchanged on exit. */

/*  X      - DOUBLE PRECISION array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise. */
/*           Before entry, the incremented array X must contain the */
/*           vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  BETA   - DOUBLE PRECISION. */
/*           On entry, BETA specifies the scalar beta. When BETA is */
/*           supplied as zero then Y need not be set on input. */
/*           Unchanged on exit. */

/*  Y      - DOUBLE PRECISION array of DIMENSION at least */
/*           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise. */
/*           Before entry, the incremented array Y must contain the */
/*           vector y. On exit, Y is overwritten by the updated vector y. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */


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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! lsame_(trans, "N", (ftnlen)1, (ftnlen)1) && ! lsame_(trans, "T", (
	    ftnlen)1, (ftnlen)1) && ! lsame_(trans, "C", (ftnlen)1, (ftnlen)1)
	    ) {
	info = 1;
    } else if (*m < 0) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*kl < 0) {
	info = 4;
    } else if (*ku < 0) {
	info = 5;
    } else if (*lda < *kl + *ku + 1) {
	info = 8;
    } else if (*incx == 0) {
	info = 10;
    } else if (*incy == 0) {
	info = 13;
    }
    if (info != 0) {
	xerbla_("DGBMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || (*alpha == 0. && *beta == 1.)) {
	return 0;
    }

/*     Set  LENX  and  LENY, the lengths of the vectors x and y, and set */
/*     up the start points in  X  and  Y. */

    if (lsame_(trans, "N", (ftnlen)1, (ftnlen)1)) {
	lenx = *n;
	leny = *m;
    } else {
	lenx = *m;
	leny = *n;
    }
    if (*incx > 0) {
	kx = 1;
    } else {
	kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
	ky = 1;
    } else {
	ky = 1 - (leny - 1) * *incy;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through the band part of A. */

/*     First form  y := beta*y. */

    if (*beta != 1.) {
	if (*incy == 1) {
	    if (*beta == 0.) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = 0.;
/* L10: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = *beta * y[i__];
/* L20: */
		}
	    }
	} else {
	    iy = ky;
	    if (*beta == 0.) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = 0.;
		    iy += *incy;
/* L30: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = *beta * y[iy];
		    iy += *incy;
/* L40: */
		}
	    }
	}
    }
    if (*alpha == 0.) {
	return 0;
    }
    kup1 = *ku + 1;
    if (lsame_(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  y := alpha*A*x + y. */

	jx = kx;
	if (*incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0.) {
		    temp = *alpha * x[jx];
		    k = kup1 - j;
/* Computing MAX */
		    i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__4 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			y[i__] += temp * a[k + i__ + j * a_dim1];
/* L50: */
		    }
		}
		jx += *incx;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0.) {
		    temp = *alpha * x[jx];
		    iy = ky;
		    k = kup1 - j;
/* Computing MAX */
		    i__4 = 1, i__2 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__3 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			y[iy] += temp * a[k + i__ + j * a_dim1];
			iy += *incy;
/* L70: */
		    }
		}
		jx += *incx;
		if (j > *ku) {
		    ky += *incy;
		}
/* L80: */
	    }
	}
    } else {

/*        Form  y := alpha*A'*x + y. */

	jy = ky;
	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp = 0.;
		k = kup1 - j;
/* Computing MAX */
		i__3 = 1, i__4 = j - *ku;
/* Computing MIN */
		i__5 = *m, i__6 = j + *kl;
		i__2 = f2c_min(i__5,i__6);
		for (i__ = f2c_max(i__3,i__4); i__ <= i__2; ++i__) {
		    temp += a[k + i__ + j * a_dim1] * x[i__];
/* L90: */
		}
		y[jy] += *alpha * temp;
		jy += *incy;
/* L100: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp = 0.;
		ix = kx;
		k = kup1 - j;
/* Computing MAX */
		i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		i__5 = *m, i__6 = j + *kl;
		i__4 = f2c_min(i__5,i__6);
		for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
		    temp += a[k + i__ + j * a_dim1] * x[ix];
		    ix += *incx;
/* L110: */
		}
		y[jy] += *alpha * temp;
		jy += *incy;
		if (j > *ku) {
		    kx += *incx;
		}
/* L120: */
	    }
	}
    }

    return 0;

/*     End of DGBMV . */

} /* dgbmv_ */

/* sgbmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(s,gbmv)(char *trans, integer *m, integer *n, integer *kl, 
	integer *ku, real *alpha, real *a, integer *lda, real *x, integer *
	incx, real *beta, real *y, integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;

    /* Local variables */
    integer info;
    real temp;
    integer lenx, leny, i__, j, k;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    integer ix, iy, jx, jy, kx, ky;
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);
    integer kup1;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SGBMV  performs one of the matrix-vector operations */

/*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y, */

/*  where alpha and beta are scalars, x and y are vectors and A is an */
/*  m by n band matrix, with kl sub-diagonals and ku super-diagonals. */

/*  Parameters */
/*  ========== */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y. */

/*              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y. */

/*              TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y. */

/*           Unchanged on exit. */

/*  M      - INTEGER. */
/*           On entry, M specifies the number of rows of the matrix A. */
/*           M must be at least zero. */
/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the number of columns of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  KL     - INTEGER. */
/*           On entry, KL specifies the number of sub-diagonals of the */
/*           matrix A. KL must satisfy  0 .le. KL. */
/*           Unchanged on exit. */

/*  KU     - INTEGER. */
/*           On entry, KU specifies the number of super-diagonals of the */
/*           matrix A. KU must satisfy  0 .le. KU. */
/*           Unchanged on exit. */

/*  ALPHA  - REAL            . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - REAL             array of DIMENSION ( LDA, n ). */
/*           Before entry, the leading ( kl + ku + 1 ) by n part of the */
/*           array A must contain the matrix of coefficients, supplied */
/*           column by column, with the leading diagonal of the matrix in */
/*           row ( ku + 1 ) of the array, the first super-diagonal */
/*           starting at position 2 in row ku, the first sub-diagonal */
/*           starting at position 1 in row ( ku + 2 ), and so on. */
/*           Elements in the array A that do not correspond to elements */
/*           in the band matrix (such as the top left ku by ku triangle) */
/*           are not referenced. */
/*           The following program segment will transfer a band matrix */
/*           from conventional full matrix storage to band storage: */

/*                 DO 20, J = 1, N */
/*                    K = KU + 1 - J */
/*                    DO 10, I = MAX( 1, J - KU ), MIN( M, J + KL ) */
/*                       A( K + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( kl + ku + 1 ). */
/*           Unchanged on exit. */

/*  X      - REAL             array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise. */
/*           Before entry, the incremented array X must contain the */
/*           vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  BETA   - REAL            . */
/*           On entry, BETA specifies the scalar beta. When BETA is */
/*           supplied as zero then Y need not be set on input. */
/*           Unchanged on exit. */

/*  Y      - REAL             array of DIMENSION at least */
/*           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise. */
/*           Before entry, the incremented array Y must contain the */
/*           vector y. On exit, Y is overwritten by the updated vector y. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */


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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! lsame_(trans, "N", (ftnlen)1, (ftnlen)1) && ! lsame_(trans, "T", (
	    ftnlen)1, (ftnlen)1) && ! lsame_(trans, "C", (ftnlen)1, (ftnlen)1)
	    ) {
	info = 1;
    } else if (*m < 0) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*kl < 0) {
	info = 4;
    } else if (*ku < 0) {
	info = 5;
    } else if (*lda < *kl + *ku + 1) {
	info = 8;
    } else if (*incx == 0) {
	info = 10;
    } else if (*incy == 0) {
	info = 13;
    }
    if (info != 0) {
	xerbla_("SGBMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || (*alpha == 0.f && *beta == 1.f)) {
	return 0;
    }

/*     Set  LENX  and  LENY, the lengths of the vectors x and y, and set */
/*     up the start points in  X  and  Y. */

    if (lsame_(trans, "N", (ftnlen)1, (ftnlen)1)) {
	lenx = *n;
	leny = *m;
    } else {
	lenx = *m;
	leny = *n;
    }
    if (*incx > 0) {
	kx = 1;
    } else {
	kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
	ky = 1;
    } else {
	ky = 1 - (leny - 1) * *incy;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through the band part of A. */

/*     First form  y := beta*y. */

    if (*beta != 1.f) {
	if (*incy == 1) {
	    if (*beta == 0.f) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = 0.f;
/* L10: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = *beta * y[i__];
/* L20: */
		}
	    }
	} else {
	    iy = ky;
	    if (*beta == 0.f) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = 0.f;
		    iy += *incy;
/* L30: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = *beta * y[iy];
		    iy += *incy;
/* L40: */
		}
	    }
	}
    }
    if (*alpha == 0.f) {
	return 0;
    }
    kup1 = *ku + 1;
    if (lsame_(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  y := alpha*A*x + y. */

	jx = kx;
	if (*incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0.f) {
		    temp = *alpha * x[jx];
		    k = kup1 - j;
/* Computing MAX */
		    i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__4 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			y[i__] += temp * a[k + i__ + j * a_dim1];
/* L50: */
		    }
		}
		jx += *incx;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0.f) {
		    temp = *alpha * x[jx];
		    iy = ky;
		    k = kup1 - j;
/* Computing MAX */
		    i__4 = 1, i__2 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__3 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			y[iy] += temp * a[k + i__ + j * a_dim1];
			iy += *incy;
/* L70: */
		    }
		}
		jx += *incx;
		if (j > *ku) {
		    ky += *incy;
		}
/* L80: */
	    }
	}
    } else {

/*        Form  y := alpha*A'*x + y. */

	jy = ky;
	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp = 0.f;
		k = kup1 - j;
/* Computing MAX */
		i__3 = 1, i__4 = j - *ku;
/* Computing MIN */
		i__5 = *m, i__6 = j + *kl;
		i__2 = f2c_min(i__5,i__6);
		for (i__ = f2c_max(i__3,i__4); i__ <= i__2; ++i__) {
		    temp += a[k + i__ + j * a_dim1] * x[i__];
/* L90: */
		}
		y[jy] += *alpha * temp;
		jy += *incy;
/* L100: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp = 0.f;
		ix = kx;
		k = kup1 - j;
/* Computing MAX */
		i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		i__5 = *m, i__6 = j + *kl;
		i__4 = f2c_min(i__5,i__6);
		for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
		    temp += a[k + i__ + j * a_dim1] * x[ix];
		    ix += *incx;
/* L110: */
		}
		y[jy] += *alpha * temp;
		jy += *incy;
		if (j > *ku) {
		    kx += *incx;
		}
/* L120: */
	    }
	}
    }

    return 0;

/*     End of SGBMV . */

} /* sgbmv_ */

/* zgbmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(z,gbmv)(char *trans, integer *m, integer *n, integer *kl, 
	integer *ku, doublecomplex *alpha, doublecomplex *a, integer *lda, 
	doublecomplex *x, integer *incx, doublecomplex *beta, doublecomplex *
	y, integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    doublecomplex z__1, z__2, z__3;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer info;
    doublecomplex temp;
    integer lenx, leny, i__, j, k;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    integer ix, iy, jx, jy, kx, ky;
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);
    logical noconj;
    integer kup1;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZGBMV  performs one of the matrix-vector operations */

/*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,   or */

/*     y := alpha*conjg( A' )*x + beta*y, */

/*  where alpha and beta are scalars, x and y are vectors and A is an */
/*  m by n band matrix, with kl sub-diagonals and ku super-diagonals. */

/*  Parameters */
/*  ========== */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y. */

/*              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y. */

/*              TRANS = 'C' or 'c'   y := alpha*conjg( A' )*x + beta*y. */

/*           Unchanged on exit. */

/*  M      - INTEGER. */
/*           On entry, M specifies the number of rows of the matrix A. */
/*           M must be at least zero. */
/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the number of columns of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  KL     - INTEGER. */
/*           On entry, KL specifies the number of sub-diagonals of the */
/*           matrix A. KL must satisfy  0 .le. KL. */
/*           Unchanged on exit. */

/*  KU     - INTEGER. */
/*           On entry, KU specifies the number of super-diagonals of the */
/*           matrix A. KU must satisfy  0 .le. KU. */
/*           Unchanged on exit. */

/*  ALPHA  - COMPLEX*16      . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - COMPLEX*16       array of DIMENSION ( LDA, n ). */
/*           Before entry, the leading ( kl + ku + 1 ) by n part of the */
/*           array A must contain the matrix of coefficients, supplied */
/*           column by column, with the leading diagonal of the matrix in */
/*           row ( ku + 1 ) of the array, the first super-diagonal */
/*           starting at position 2 in row ku, the first sub-diagonal */
/*           starting at position 1 in row ( ku + 2 ), and so on. */
/*           Elements in the array A that do not correspond to elements */
/*           in the band matrix (such as the top left ku by ku triangle) */
/*           are not referenced. */
/*           The following program segment will transfer a band matrix */
/*           from conventional full matrix storage to band storage: */

/*                 DO 20, J = 1, N */
/*                    K = KU + 1 - J */
/*                    DO 10, I = MAX( 1, J - KU ), MIN( M, J + KL ) */
/*                       A( K + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( kl + ku + 1 ). */
/*           Unchanged on exit. */

/*  X      - COMPLEX*16       array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise. */
/*           Before entry, the incremented array X must contain the */
/*           vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  BETA   - COMPLEX*16      . */
/*           On entry, BETA specifies the scalar beta. When BETA is */
/*           supplied as zero then Y need not be set on input. */
/*           Unchanged on exit. */

/*  Y      - COMPLEX*16       array of DIMENSION at least */
/*           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n' */
/*           and at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise. */
/*           Before entry, the incremented array Y must contain the */
/*           vector y. On exit, Y is overwritten by the updated vector y. */


/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */


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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (! lsame_(trans, "N", (ftnlen)1, (ftnlen)1) && ! lsame_(trans, "T", (
	    ftnlen)1, (ftnlen)1) && ! lsame_(trans, "C", (ftnlen)1, (ftnlen)1)
	    ) {
	info = 1;
    } else if (*m < 0) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*kl < 0) {
	info = 4;
    } else if (*ku < 0) {
	info = 5;
    } else if (*lda < *kl + *ku + 1) {
	info = 8;
    } else if (*incx == 0) {
	info = 10;
    } else if (*incy == 0) {
	info = 13;
    }
    if (info != 0) {
	xerbla_("ZGBMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0 || (alpha->r == 0. && alpha->i == 0. && (beta->r == 
	    1. && beta->i == 0.))) {
	return 0;
    }

    noconj = lsame_(trans, "T", (ftnlen)1, (ftnlen)1);

/*     Set  LENX  and  LENY, the lengths of the vectors x and y, and set */
/*     up the start points in  X  and  Y. */

    if (lsame_(trans, "N", (ftnlen)1, (ftnlen)1)) {
	lenx = *n;
	leny = *m;
    } else {
	lenx = *m;
	leny = *n;
    }
    if (*incx > 0) {
	kx = 1;
    } else {
	kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
	ky = 1;
    } else {
	ky = 1 - (leny - 1) * *incy;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through the band part of A. */

/*     First form  y := beta*y. */

    if (beta->r != 1. || beta->i != 0.) {
	if (*incy == 1) {
	    if (beta->r == 0. && beta->i == 0.) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    y[i__2].r = 0., y[i__2].i = 0.;
/* L10: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    i__3 = i__;
		    z__1.r = beta->r * y[i__3].r - beta->i * y[i__3].i, 
			    z__1.i = beta->r * y[i__3].i + beta->i * y[i__3]
			    .r;
		    y[i__2].r = z__1.r, y[i__2].i = z__1.i;
/* L20: */
		}
	    }
	} else {
	    iy = ky;
	    if (beta->r == 0. && beta->i == 0.) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = iy;
		    y[i__2].r = 0., y[i__2].i = 0.;
		    iy += *incy;
/* L30: */
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = iy;
		    i__3 = iy;
		    z__1.r = beta->r * y[i__3].r - beta->i * y[i__3].i, 
			    z__1.i = beta->r * y[i__3].i + beta->i * y[i__3]
			    .r;
		    y[i__2].r = z__1.r, y[i__2].i = z__1.i;
		    iy += *incy;
/* L40: */
		}
	    }
	}
    }
    if (alpha->r == 0. && alpha->i == 0.) {
	return 0;
    }
    kup1 = *ku + 1;
    if (lsame_(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*        Form  y := alpha*A*x + y. */

	jx = kx;
	if (*incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		if (x[i__2].r != 0. || x[i__2].i != 0.) {
		    i__2 = jx;
		    z__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    z__1.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    temp.r = z__1.r, temp.i = z__1.i;
		    k = kup1 - j;
/* Computing MAX */
		    i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__4 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			i__2 = i__;
			i__3 = i__;
			i__5 = k + i__ + j * a_dim1;
			z__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i, 
				z__2.i = temp.r * a[i__5].i + temp.i * a[i__5]
				.r;
			z__1.r = y[i__3].r + z__2.r, z__1.i = y[i__3].i + 
				z__2.i;
			y[i__2].r = z__1.r, y[i__2].i = z__1.i;
/* L50: */
		    }
		}
		jx += *incx;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__4 = jx;
		if (x[i__4].r != 0. || x[i__4].i != 0.) {
		    i__4 = jx;
		    z__1.r = alpha->r * x[i__4].r - alpha->i * x[i__4].i, 
			    z__1.i = alpha->r * x[i__4].i + alpha->i * x[i__4]
			    .r;
		    temp.r = z__1.r, temp.i = z__1.i;
		    iy = ky;
		    k = kup1 - j;
/* Computing MAX */
		    i__4 = 1, i__2 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__3 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			i__4 = iy;
			i__2 = iy;
			i__5 = k + i__ + j * a_dim1;
			z__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i, 
				z__2.i = temp.r * a[i__5].i + temp.i * a[i__5]
				.r;
			z__1.r = y[i__2].r + z__2.r, z__1.i = y[i__2].i + 
				z__2.i;
			y[i__4].r = z__1.r, y[i__4].i = z__1.i;
			iy += *incy;
/* L70: */
		    }
		}
		jx += *incx;
		if (j > *ku) {
		    ky += *incy;
		}
/* L80: */
	    }
	}
    } else {

/*        Form  y := alpha*A'*x + y  or  y := alpha*conjg( A' )*x + y. */

	jy = ky;
	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp.r = 0., temp.i = 0.;
		k = kup1 - j;
		if (noconj) {
/* Computing MAX */
		    i__3 = 1, i__4 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__2 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__3,i__4); i__ <= i__2; ++i__) {
			i__3 = k + i__ + j * a_dim1;
			i__4 = i__;
			z__2.r = a[i__3].r * x[i__4].r - a[i__3].i * x[i__4]
				.i, z__2.i = a[i__3].r * x[i__4].i + a[i__3]
				.i * x[i__4].r;
			z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
			temp.r = z__1.r, temp.i = z__1.i;
/* L90: */
		    }
		} else {
/* Computing MAX */
		    i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__4 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__2,i__3); i__ <= i__4; ++i__) {
			d_cnjg(&z__3, &a[k + i__ + j * a_dim1]);
			i__2 = i__;
			z__2.r = z__3.r * x[i__2].r - z__3.i * x[i__2].i, 
				z__2.i = z__3.r * x[i__2].i + z__3.i * x[i__2]
				.r;
			z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
			temp.r = z__1.r, temp.i = z__1.i;
/* L100: */
		    }
		}
		i__4 = jy;
		i__2 = jy;
		z__2.r = alpha->r * temp.r - alpha->i * temp.i, z__2.i = 
			alpha->r * temp.i + alpha->i * temp.r;
		z__1.r = y[i__2].r + z__2.r, z__1.i = y[i__2].i + z__2.i;
		y[i__4].r = z__1.r, y[i__4].i = z__1.i;
		jy += *incy;
/* L110: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp.r = 0., temp.i = 0.;
		ix = kx;
		k = kup1 - j;
		if (noconj) {
/* Computing MAX */
		    i__4 = 1, i__2 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__3 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__4,i__2); i__ <= i__3; ++i__) {
			i__4 = k + i__ + j * a_dim1;
			i__2 = ix;
			z__2.r = a[i__4].r * x[i__2].r - a[i__4].i * x[i__2]
				.i, z__2.i = a[i__4].r * x[i__2].i + a[i__4]
				.i * x[i__2].r;
			z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
			temp.r = z__1.r, temp.i = z__1.i;
			ix += *incx;
/* L120: */
		    }
		} else {
/* Computing MAX */
		    i__3 = 1, i__4 = j - *ku;
/* Computing MIN */
		    i__5 = *m, i__6 = j + *kl;
		    i__2 = f2c_min(i__5,i__6);
		    for (i__ = f2c_max(i__3,i__4); i__ <= i__2; ++i__) {
			d_cnjg(&z__3, &a[k + i__ + j * a_dim1]);
			i__3 = ix;
			z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i, 
				z__2.i = z__3.r * x[i__3].i + z__3.i * x[i__3]
				.r;
			z__1.r = temp.r + z__2.r, z__1.i = temp.i + z__2.i;
			temp.r = z__1.r, temp.i = z__1.i;
			ix += *incx;
/* L130: */
		    }
		}
		i__2 = jy;
		i__3 = jy;
		z__2.r = alpha->r * temp.r - alpha->i * temp.i, z__2.i = 
			alpha->r * temp.i + alpha->i * temp.r;
		z__1.r = y[i__3].r + z__2.r, z__1.i = y[i__3].i + z__2.i;
		y[i__2].r = z__1.r, y[i__2].i = z__1.i;
		jy += *incy;
		if (j > *ku) {
		    kx += *incx;
		}
/* L140: */
	    }
	}
    }

    return 0;

/*     End of ZGBMV . */

} /* zgbmv_ */

#endif

