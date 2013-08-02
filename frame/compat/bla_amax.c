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
#undef  GENTFUNC2I
#define GENTFUNC2I( ftype_x, ftype_i, chx, chi, blasname, blisname ) \
\
ftype_i PASTEF772(chi,chx,blasname)( \
                                     fint*    n, \
                                     ftype_x* x, fint* incx  \
                                   ) \
{ \
	dim_t    n0; \
	ftype_x* x0; \
	inc_t    incx0; \
	ftype_i  index; \
\
	/* Convert negative values of n to zero. */ \
	bli_convert_blas_dim1( *n, n0 ); \
\
	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */ \
	bli_convert_blas_incv( n0, x, *incx, x0, incx0 ); \
\
	/* Call BLIS interface. */ \
	PASTEMAC(chx,abmaxv)( n0, \
	                      x0, incx0, \
	                      &index ); \
\
	/* Convert zero-based BLIS (C) index to one-based BLAS (Fortran)
	   index. */ \
	index++; \
\
	return index; \
}

#ifdef BLIS_ENABLE_BLAS2BLIS
INSERT_GENTFUNC2I_BLAS( amax, abmaxv )
#endif

