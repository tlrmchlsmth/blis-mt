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



void bli_strsm_u_opt_mxn(
                          float*    restrict a,
                          float*    restrict b,
                          float*    restrict bd,
                          float*    restrict c, inc_t rs_c, inc_t cs_c
                        )
{
    /* Just call the reference implementation. */
    bli_strsm_u_ref_mxn( a,
                         b,
                         bd,
                         c, rs_c, cs_c );
}



void bli_dtrsm_u_opt_mxn(
                          double*   restrict a,
                          double*   restrict b,
                          double*   restrict bd,
                          double*   restrict c, inc_t rs_c, inc_t cs_c
                        )
{
/*
  Template trsm_u micro-kernel implementation

  This function contains a template implementation for a double-precision
  real trsm micro-kernel, coded in C, which can serve as the starting point
  for one to write an optimized micro-kernel on an arbitrary architecture.
  (We show a template implementation for only double-precision real because
  the templates for the other three floating-point types would be nearly
  identical.)

  This micro-kernel performs a triangular solve with NR right-hand sides:

    C := inv(A) * B

  where A is MR x MR and upper triangular, B is MR x NR, and C is MR x NR.

  NOTE: Here, this trsm micro-kernel supports element "duplication", a
  feature that is enabled or disabled in bli_kernel.h. Duplication factors
  are also defined in the aforementioned header. Duplication is NOT
  commonly used and most developers may assume it is disabled.

  Parameters:

  - a:      The address of A, which is the MR x MR upper triangular block
            within the packed (column-stored) micro-panel of A. By the time
            this trsm micro-kernel is called, the diagonal of A has already
            been inverted and the strictly lower triangle contains zeros.
  - b:      The address of B, which is the MR x NR subpartition of the
            current packed (row-stored) micro-panel of B.
  - bd:     The address of the duplicated copy of B. If duplication is
            disabled, then bd == b.
  - c:      The address of C, which is the MR x NR block of the output
            matrix (ie: the matrix provided by the user to the highest-level
            trsm API call). C corresponds to the elements that exist in
            packed form in B, and is stored according to rs_c and cs_c.
  - rs_c:   The row stride of C (ie: the distance to the next row of C11,
            in units of matrix elements).
  - cs_c:   The column stride of C (ie: the distance to the next column of
            C11, in units of matrix elements).

  Please see the comments in bli_gemmtrsm_u_opt_mxn.c for a diagram of the
  trsm operation and where it fits in with the preceding gemm subproblem.

  Here are a few things to consider:
  - While all three loops are exposed in this template micro-kernel, all
    three loops typically disappear in an optimized code because they are
    fully unrolled.
  - Note that the diagonal of the triangular matrix A contains the INVERSE
    of those elements. This is done during packing so that we can avoid
    expensive division instructions within this micro-kernel.
  - This micro-kernel assumes duplication is NOT enabled. If it IS enabled,
    then the result must be written to three places: the sub-block within the
    duplicated copy of B, the sub-block of the original packed micro-panel of
    B, and the sub-block of the output matrix C. When duplication is not
    used, the micro-kernel should update only the latter two locations.

  For more info, please refer to the BLIS website and/or contact the
  blis-devel mailing list.

  -FGVZ
*/
	const dim_t        m     = bli_dmr;
	const dim_t        n     = bli_dnr;

	const inc_t        rs_a  = 1;
	const inc_t        cs_a  = bli_dpackmr;

	const inc_t        rs_b  = bli_dpacknr;
	const inc_t        cs_b  = 1;

	dim_t              iter, i, j, l;
	dim_t              n_behind;

	double*   restrict alpha11;
	double*   restrict a12t;
	double*   restrict alpha12;
	double*   restrict X2;
	double*   restrict x1;
	double*   restrict x21;
	double*   restrict chi21;
	double*   restrict chi11;
	double*   restrict gamma11;
	double             rho11;

	for ( iter = 0; iter < m; ++iter )
	{
		i        = m - iter - 1;
		n_behind = iter;
		alpha11  = a + (i  )*rs_a + (i  )*cs_a;
		a12t     = a + (i  )*rs_a + (i+1)*cs_a;
		x1       = b + (i  )*rs_b + (0  )*cs_b;
		X2       = b + (i+1)*rs_b + (0  )*cs_b;

		/* x1 = x1 - a12t * X2; */
		/* x1 = x1 / alpha11; */
		for ( j = 0; j < n; ++j )
		{
			chi11   = x1 + (0  )*rs_b + (j  )*cs_b;
			x21     = X2 + (0  )*rs_b + (j  )*cs_b;
			gamma11 = c  + (i  )*rs_c + (j  )*cs_c;

			/* chi11 = chi11 - a12t * x21; */
			bli_dset0s( rho11 );
			for ( l = 0; l < n_behind; ++l )
			{
				alpha12 = a12t + (l  )*cs_a;
				chi21   = x21  + (l  )*rs_b;

				bli_daxpys( *alpha12, *chi21, rho11 );
			}
			bli_dsubs( rho11, *chi11 );

			/* chi11 = chi11 / alpha11; */
			/* NOTE: The INVERSE of alpha11 (1.0/alpha11) is stored instead
			   of alpha11, so we can multiply rather than divide. We store 
			   the inverse of alpha11 intentionally to avoid expensive
			   division instructions within the micro-kernel. */
			bli_dscals( *alpha11, *chi11 );

			/* Output final result to matrix C. */
			bli_dcopys( *chi11, *gamma11 );
		}
	}
}



void bli_ctrsm_u_opt_mxn(
                          scomplex* restrict a,
                          scomplex* restrict b,
                          scomplex* restrict bd,
                          scomplex* restrict c, inc_t rs_c, inc_t cs_c
                        )
{
    /* Just call the reference implementation. */
    bli_ctrsm_u_ref_mxn( a,
                         b,
                         bd,
                         c, rs_c, cs_c );
}



void bli_ztrsm_u_opt_mxn(
                          dcomplex* restrict a,
                          dcomplex* restrict b,
                          dcomplex* restrict bd,
                          dcomplex* restrict c, inc_t rs_c, inc_t cs_c
                        )
{
    /* Just call the reference implementation. */
    bli_ztrsm_u_ref_mxn( a,
                         b,
                         bd,
                         c, rs_c, cs_c );
}

