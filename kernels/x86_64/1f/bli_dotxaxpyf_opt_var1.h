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


// Define fusing factors.
#define bli_sdotxaxpyf_fuse_fac   ( BLIS_DEFAULT_FUSING_FACTOR_S     )
#define bli_ddotxaxpyf_fuse_fac   ( BLIS_DEFAULT_FUSING_FACTOR_D     )
#define bli_cdotxaxpyf_fuse_fac   ( BLIS_DEFAULT_FUSING_FACTOR_C / 2 )
#define bli_zdotxaxpyf_fuse_fac   ( BLIS_DEFAULT_FUSING_FACTOR_Z / 2 )


#undef  GENTPROT3U12
#define GENTPROT3U12( ctype_a, ctype_b, ctype_c, ctype_ab, cha, chb, chc, chab, varname ) \
\
void PASTEMAC3(cha,chb,chc,varname)( \
                                     conj_t conjat, \
                                     conj_t conja, \
                                     conj_t conjw, \
                                     conj_t conjx, \
                                     dim_t  m, \
                                     dim_t  n, \
                                     void*  alpha, \
                                     void*  a, inc_t inca, inc_t lda, \
                                     void*  w, inc_t incw, \
                                     void*  x, inc_t incx, \
                                     void*  beta, \
                                     void*  y, inc_t incy, \
                                     void*  z, inc_t incz \
                                   );

INSERT_GENTPROT3U12_BASIC( dotxaxpyf_opt_var1 )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTPROT3U12_MIX_D( dotxaxpyf_opt_var1 )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTPROT3U12_MIX_P( dotxaxpyf_opt_var1 )
#endif

