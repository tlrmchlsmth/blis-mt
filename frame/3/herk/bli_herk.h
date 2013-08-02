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

#include "bli_herk_cntl.h"
#include "bli_herk_check.h"
#include "bli_herk_int.h"

#include "bli_herk_l_blk_var1.h"
#include "bli_herk_u_blk_var1.h"

#include "bli_herk_l_blk_var2.h"
#include "bli_herk_u_blk_var2.h"

#include "bli_herk_blk_var3.h"

#include "bli_herk_l_blk_var4.h"
#include "bli_herk_u_blk_var4.h"

#include "bli_herk_l_ker_var2.h"
#include "bli_herk_u_ker_var2.h"


//
// Prototype object-based interface.
//
void bli_herk( obj_t*  alpha,
               obj_t*  a,
               obj_t*  beta,
               obj_t*  c );


//
// Prototype BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname)( \
                          uplo_t    uploc, \
                          trans_t   transa, \
                          dim_t     m, \
                          dim_t     k, \
                          ctype_r*  alpha, \
                          ctype*    a, inc_t rs_a, inc_t cs_a, \
                          ctype_r*  beta, \
                          ctype*    c, inc_t rs_c, inc_t cs_c  \
                        );

INSERT_GENTPROTR_BASIC( herk )


//
// Prototype BLAS-like interfaces with heterogeneous-typed operands.
//
#undef  GENTPROT2R
#define GENTPROT2R( ctype_a, ctype_c, ctype_ar, cha, chc, chAr, opname ) \
\
void PASTEMAC2(cha,chc,opname)( \
                                uplo_t    uploc, \
                                trans_t   transa, \
                                dim_t     m, \
                                dim_t     k, \
                                ctype_ar* alpha, \
                                ctype_a*  a, inc_t rs_a, inc_t cs_a, \
                                ctype_c*  beta, \
                                ctype_c*  c, inc_t rs_c, inc_t cs_c  \
                              );

INSERT_GENTPROT2R_BASIC( herk )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTPROT2R_MIX_D( herk )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTPROT2R_MIX_P( herk )
#endif

