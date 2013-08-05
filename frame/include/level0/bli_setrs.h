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

#ifndef BLIS_SETRS_H
#define BLIS_SETRS_H

// setrs

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.


#define bli_sssetrs( xr, y ) \
{ \
	(y)      = ( float  ) (xr); \
}
#define bli_dssetrs( xr, y ) \
{ \
	(y)      = ( float  ) (xr); \
}


#define bli_sdsetrs( xr, y ) \
{ \
	(y)      = ( double ) (xr); \
}
#define bli_ddsetrs( xr, y ) \
{ \
	(y)      = ( double ) (xr); \
}


#ifndef BLIS_ENABLE_C99_COMPLEX


#define bli_scsetrs( xr, y ) \
{ \
	bli_creal(y) = ( float  ) (xr); \
}
#define bli_dcsetrs( xr, y ) \
{ \
	bli_creal(y) = ( float  ) (xr); \
}


#define bli_szsetrs( xr, y ) \
{ \
	bli_zreal(y) = ( double ) (xr); \
}
#define bli_dzsetrs( xr, y ) \
{ \
	bli_zreal(y) = ( double ) (xr); \
}


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_scsetrs( xr, y ) \
{ \
	(y)      = ( float  ) (xr) + bli_cimag(y) * (I); \
}
#define bli_dcsetrs( xr, y ) \
{ \
	(y)      = ( float  ) (xr) + bli_cimag(y) * (I); \
}


#define bli_szsetrs( xr, y ) \
{ \
	(y)      = ( double ) (xr) + bli_zimag(y) * (I); \
}
#define bli_dzsetrs( xr, y ) \
{ \
	(y)      = ( double ) (xr) + bli_zimag(y) * (I); \
}


#endif // BLIS_ENABLE_C99_COMPLEX


#define bli_ssetrs( xr, y )  bli_sssetrs( xr, y )
#define bli_dsetrs( xr, y )  bli_ddsetrs( xr, y )
#define bli_csetrs( xr, y )  bli_scsetrs( xr, y )
#define bli_zsetrs( xr, y )  bli_dzsetrs( xr, y )


#endif
