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


//
// Define BLAS-to-BLIS interfaces.
//
#undef  GENTFUNCSCAL
#define GENTFUNCSCAL( ftype_a, ftype_x, cha, chx, blasname, blisname ) \
\
void PASTEF772(chx,cha,blasname)( \
                                  f77_int* n, \
                                  ftype_a* alpha, \
                                  ftype_x* x, f77_int* incx  \
                                ) \
{ \
	dim_t    n0; \
	ftype_x* x0; \
	inc_t    incx0; \
	ftype_x  alpha_cast; \
	err_t    init_result; \
\
	/* Initialize BLIS (if it is not already initialized). */ \
	bli_init_safe( &init_result ); \
\
	/* Convert negative values of n to zero. */ \
	bli_convert_blas_dim1( *n, n0 ); \
\
	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */ \
	bli_convert_blas_incv( n0, x, *incx, x0, incx0 ); \
\
	/* NOTE: We do not natively implement BLAS's csscal/zdscal in BLIS
	   UNLESS mixed domain functionality is enabled at configure-time.
	   However, we don't want to assume that BLIS was configured that
	   way, so we will just always sub-optimally implement those cases
	   by casting alpha to ctype_x (potentially the complex domain) and
	   using the homogeneous datatype instance according to that type. */ \
	PASTEMAC2(cha,chx,cast)( alpha, alpha_cast ); \
\
	/* Call BLIS interface. */ \
	PASTEMAC2(chx,chx,blisname)( BLIS_NO_CONJUGATE, \
	                             n0, \
	                             &alpha_cast, \
	                             x0, incx0 ); \
\
	/* Finalize BLIS (if it was initialized above). */ \
	bli_finalize_safe( init_result ); \
}

#ifdef BLIS_ENABLE_BLAS2BLIS
INSERT_GENTFUNCSCAL_BLAS( scal, SCALV_KERNEL )
#endif

