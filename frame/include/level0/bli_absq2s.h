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

#ifndef BLIS_ABSQR2_H
#define BLIS_ABSQR2_H

// absq2s

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of a.


#define bli_ssabsq2s( x, a ) \
{ \
	(a) = ( float  )( (x)      * (x) ); \
}
#define bli_dsabsq2s( x, a ) \
{ \
	(a) = ( float  )( (x)      * (x) ); \
}
#define bli_csabsq2s( x, a ) \
{ \
	(a) = ( float  )( (x).real * (x).real + \
	                  (x).imag * (x).imag ); \
}
#define bli_zsabsq2s( x, a ) \
{ \
	(a) = ( float  )( (x).real * (x).real + \
	                  (x).imag * (x).imag ); \
}


#define bli_sdabsq2s( x, a ) \
{ \
	(a) = ( double )( (x)      * (x) ); \
}
#define bli_ddabsq2s( x, a ) \
{ \
	(a) = ( double )( (x)      * (x) ); \
}
#define bli_cdabsq2s( x, a ) \
{ \
	(a) = ( double )( (x).real * (x).real + \
	                  (x).imag * (x).imag ); \
}
#define bli_zdabsq2s( x, a ) \
{ \
	(a) = ( double )( (x).real * (x).real + \
	                  (x).imag * (x).imag ); \
}


#define bli_scabsq2s( x, a ) \
{ \
	(a).real = ( float  )( (x)      * (x) ); \
	(a).imag = 0.0F; \
}
#define bli_dcabsq2s( x, a ) \
{ \
	(a).real = ( float  )( (x)      * (x) ); \
	(a).imag = 0.0; \
}
#define bli_ccabsq2s( x, a ) \
{ \
	(a).real = ( float  )( (x).real * (x).real + \
	                       (x).imag * (x).imag ); \
	(a).imag = 0.0F; \
}
#define bli_zcabsq2s( x, a ) \
{ \
	(a).real = ( float  )( (x).real * (x).real + \
	                       (x).imag * (x).imag ); \
	(a).imag = 0.0; \
}


#define bli_szabsq2s( x, a ) \
{ \
	(a).real = ( double )( (x)      * (x) ); \
	(a).imag = 0.0F; \
}
#define bli_dzabsq2s( x, a ) \
{ \
	(a).real = ( double )( (x)      * (x) ); \
	(a).imag = 0.0; \
}
#define bli_czabsq2s( x, a ) \
{ \
	(a).real = ( double )( (x).real * (x).real + \
	                       (x).imag * (x).imag ); \
	(a).imag = 0.0F; \
}
#define bli_zzabsq2s( x, a ) \
{ \
	(a).real = ( double )( (x).real * (x).real + \
	                       (x).imag * (x).imag ); \
	(a).imag = 0.0; \
}


#define bli_sabsq2s( x, a ) \
{ \
	bli_ssabsq2s( x, a ); \
}
#define bli_dabsq2s( x, a ) \
{ \
	bli_ddabsq2s( x, a ); \
}
#define bli_cabsq2s( x, a ) \
{ \
	bli_ccabsq2s( x, a ); \
}
#define bli_zabsq2s( x, a ) \
{ \
	bli_zzabsq2s( x, a ); \
}


#endif
