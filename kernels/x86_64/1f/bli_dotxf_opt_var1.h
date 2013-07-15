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
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

void bli_dotxf_opt_var1( obj_t* alpha,
                         obj_t* x,
                         obj_t* y,
                         obj_t* beta,
                         obj_t* rho );


//
// Define fusing factors for dotxf operation.
//
#define bli_sdotxf_fuse_fac   BLIS_DEFAULT_FUSING_FACTOR_S
#define bli_ddotxf_fuse_fac   BLIS_DEFAULT_FUSING_FACTOR_D
#define bli_cdotxf_fuse_fac   BLIS_DEFAULT_FUSING_FACTOR_C
#define bli_zdotxf_fuse_fac   BLIS_DEFAULT_FUSING_FACTOR_Z


#undef  GENTPROT3U12
#define GENTPROT3U12( ctype_x, ctype_y, ctype_r, ctype_xy, chx, chy, chr, chxy, varname ) \
\
void PASTEMAC3(chx,chy,chr,varname)( \
                                     conj_t conjx, \
                                     conj_t conjy, \
                                     dim_t  m, \
                                     dim_t  n, \
                                     void*  alpha, \
                                     void*  x, inc_t incx, inc_t ldx, \
                                     void*  y, inc_t incy, \
                                     void*  beta, \
                                     void*  r, inc_t incr \
                                   );

INSERT_GENTPROT3U12_BASIC( dotxf_opt_var1 )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTPROT3U12_MIX_D( dotxf_opt_var1 )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTPROT3U12_MIX_P( dotxf_opt_var1 )
#endif

