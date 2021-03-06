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

#ifndef BLIS_ADDS_H
#define BLIS_ADDS_H

// adds

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of y.


#define bli_ssadds( a, y ) \
{ \
	(y) += bli_sreal(a); \
}
#define bli_dsadds( a, y ) \
{ \
	(y) += bli_dreal(a); \
}
#define bli_csadds( a, y ) \
{ \
	(y) += bli_creal(a); \
}
#define bli_zsadds( a, y ) \
{ \
	(y) += bli_zreal(a); \
}


#define bli_sdadds( a, y ) \
{ \
	(y) += bli_sreal(a); \
}
#define bli_ddadds( a, y ) \
{ \
	(y) += bli_dreal(a); \
}
#define bli_cdadds( a, y ) \
{ \
	(y) += bli_creal(a); \
}
#define bli_zdadds( a, y ) \
{ \
	(y) += bli_zreal(a); \
}


#ifndef BLIS_ENABLE_C99_COMPLEX


#define bli_scadds( a, y ) \
{ \
	bli_creal(y) += bli_sreal(a); \
}
#define bli_dcadds( a, y ) \
{ \
	bli_creal(y) += bli_dreal(a); \
}
#define bli_ccadds( a, y ) \
{ \
	bli_creal(y) += bli_creal(a); \
	bli_cimag(y) += bli_cimag(a); \
}
#define bli_zcadds( a, y ) \
{ \
	bli_creal(y) += bli_zreal(a); \
	bli_cimag(y) += bli_zimag(a); \
}


#define bli_szadds( a, y ) \
{ \
	bli_zreal(y) += bli_sreal(a); \
}
#define bli_dzadds( a, y ) \
{ \
	bli_zreal(y) += bli_dreal(a); \
}
#define bli_czadds( a, y ) \
{ \
	bli_zreal(y) += bli_creal(a); \
	bli_zimag(y) += bli_cimag(a); \
}
#define bli_zzadds( a, y ) \
{ \
	bli_zreal(y) += bli_zreal(a); \
	bli_zimag(y) += bli_zimag(a); \
}


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_scadds( a, y )  { (y) += (a); }
#define bli_dcadds( a, y )  { (y) += (a); }
#define bli_ccadds( a, y )  { (y) += (a); }
#define bli_zcadds( a, y )  { (y) += (a); }

#define bli_szadds( a, y )  { (y) += (a); }
#define bli_dzadds( a, y )  { (y) += (a); }
#define bli_czadds( a, y )  { (y) += (a); }
#define bli_zzadds( a, y )  { (y) += (a); }


#endif // BLIS_ENABLE_C99_COMPLEX


#define bli_sadds( a, y )  bli_ssadds( a, y )
#define bli_dadds( a, y )  bli_ddadds( a, y )
#define bli_cadds( a, y )  bli_ccadds( a, y )
#define bli_zadds( a, y )  bli_zzadds( a, y )


#endif
