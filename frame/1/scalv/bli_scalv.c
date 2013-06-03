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

/*
extern scalv_t* scalv_cntl;

void bli_scalv( obj_t* beta,
                obj_t* x )
{
	//bli_scalv_int( beta,
	//               x,
	//               scalv_cntl );
	bli_scalv_unb_var1( beta, x );
}
*/

//
// Define object-based interface.
//
#undef  GENFRONT
#define GENFRONT( opname, varname ) \
\
void PASTEMAC0(opname)( \
                        obj_t* beta, \
                        obj_t* x \
                      ) \
{ \
    if ( bli_error_checking_is_enabled() ) \
        PASTEMAC(opname,_check)( beta, x ); \
\
    PASTEMAC0(varname)( beta, \
                        x ); \
}

GENFRONT( scalv, SCALV_KERNEL )


//
// Define BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,opname)( \
                          conj_t conjbeta, \
                          dim_t  n, \
                          ctype* beta, \
                          ctype* x, inc_t incx \
                        ) \
{ \
	PASTEMAC2(ch,ch,varname)( conjbeta, \
	                          n, \
	                          beta, \
	                          x, incx ); \
}

INSERT_GENTFUNC_BASIC( scalv, SCALV_KERNEL )


//
// Define BLAS-like interfaces with heterogeneous-typed operands.
//
#undef  GENTFUNC2
#define GENTFUNC2( ctype_b, ctype_x, chb, chx, opname, varname ) \
\
void PASTEMAC2(chb,chx,opname)( \
                                conj_t   conjbeta, \
                                dim_t    n, \
                                ctype_b* beta, \
                                ctype_x* x, inc_t incx \
                              ) \
{ \
	PASTEMAC2(chb,chx,varname)( conjbeta, \
	                            n, \
	                            beta, \
	                            x, incx ); \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC2_BASIC( scalv, SCALV_KERNEL )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC2_MIX_D( scalv, SCALV_KERNEL )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC2_MIX_P( scalv, SCALV_KERNEL )
#endif

