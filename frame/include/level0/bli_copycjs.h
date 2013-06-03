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

#ifndef BLIS_COPYCJS_H
#define BLIS_COPYCJS_H

// copycjs

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.
// - x is (conditionally) copied in conjugated form.

#define bli_sscopycjs( conj, x, y ) \
{ \
	(y) = ( float  ) (x); \
}
#define bli_dscopycjs( conj, x, y ) \
{ \
	(y) = ( float  ) (x); \
}
#define bli_cscopycjs( conj, x, y ) \
{ \
	(y) = ( float  ) (x).real; \
}
#define bli_zscopycjs( conj, x, y ) \
{ \
	(y) = ( float  ) (x).real; \
}

#define bli_sdcopycjs( conj, x, y ) \
{ \
	(y) = ( double ) (x); \
}
#define bli_ddcopycjs( conj, x, y ) \
{ \
	(y) = ( double ) (x); \
}
#define bli_cdcopycjs( conj, x, y ) \
{ \
	(y) = ( double ) (x).real; \
}
#define bli_zdcopycjs( conj, x, y ) \
{ \
	(y) = ( double ) (x).real; \
}

#define bli_sccopycjs( conj, x, y ) \
{ \
	(y).real = ( float  ) (x); \
	(y).imag = 0.0F; \
}
#define bli_dccopycjs( conj, x, y ) \
{ \
	(y).real = ( float  ) (x); \
	(y).imag = 0.0F; \
}
#define bli_cccopycjs( conj, x, y ) \
{ \
	(y).real = ( float  )  (x).real; \
	if ( bli_is_conj( conj ) ) \
		(y).imag = ( float  ) -(x).imag; \
	else \
		(y).imag = ( float  )  (x).imag; \
}
#define bli_zccopycjs( conj, x, y ) \
{ \
	(y).real = ( float  )  (x).real; \
	if ( bli_is_conj( conj ) ) \
		(y).imag = ( float  ) -(x).imag; \
	else \
		(y).imag = ( float  )  (x).imag; \
}

#define bli_szcopycjs( conj, x, y ) \
{ \
	(y).real = ( double ) (x); \
	(y).imag = 0.0; \
}
#define bli_dzcopycjs( conj, x, y ) \
{ \
	(y).real = ( double ) (x); \
	(y).imag = 0.0; \
}
#define bli_czcopycjs( conj, x, y ) \
{ \
	(y).real = ( double )  (x).real; \
	if ( bli_is_conj( conj ) ) \
		(y).imag = ( double ) -(x).imag; \
	else \
		(y).imag = ( double )  (x).imag; \
}
#define bli_zzcopycjs( conj, x, y ) \
{ \
	(y).real = ( double )  (x).real; \
	if ( bli_is_conj( conj ) ) \
		(y).imag = ( double ) -(x).imag; \
	else \
		(y).imag = ( double )  (x).imag; \
}


#define bli_scopycjs( conj, x, y ) \
{ \
	bli_sscopycjs( conj, x, y ); \
}
#define bli_dcopycjs( conj, x, y ) \
{ \
	bli_ddcopycjs( conj, x, y ); \
}
#define bli_ccopycjs( conj, x, y ) \
{ \
	bli_cccopycjs( conj, x, y ); \
}
#define bli_zcopycjs( conj, x, y ) \
{ \
	bli_zzcopycjs( conj, x, y ); \
}


#endif
