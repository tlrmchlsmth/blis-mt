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

#ifndef BLIS_INVSCALS_H
#define BLIS_INVSCALS_H

// invscals

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.

#define bli_ssinvscals( a, x ) \
{ \
	(x) /= ( float  ) (a); \
}
#define bli_dsinvscals( a, x ) \
{ \
	(x) /= ( float  ) (a); \
}
#define bli_csinvscals( a, x ) \
{ \
	(x) /= ( float  ) (a).real; \
}
#define bli_zsinvscals( a, x ) \
{ \
	(x) /= ( float  ) (a).real; \
}

#define bli_sdinvscals( a, x ) \
{ \
	(x) /= ( double ) (a); \
}
#define bli_ddinvscals( a, x ) \
{ \
	(x) /= ( double ) (a); \
}
#define bli_cdinvscals( a, x ) \
{ \
	(x) /= ( double ) (a).real; \
}
#define bli_zdinvscals( a, x ) \
{ \
	(x) /= ( double ) (a).real; \
}

#define bli_scinvscals( a, x ) \
{ \
	(x).real /= ( float  ) (a); \
	(x).imag /= ( float  ) (a); \
}
#define bli_dcinvscals( a, x ) \
{ \
	(x).real /= ( float  ) (a); \
	(x).imag /= ( float  ) (a); \
}
#define bli_ccinvscals( a, x ) \
{ \
	float  temp =              ( float  ) (a).real * (a).real + ( float  ) (a).imag * (a).imag; \
	float  xr   = ( float  ) ( ( float  ) (a).real * (x).real + ( float  ) (a).imag * (x).imag ) / temp; \
	float  xi   = ( float  ) ( ( float  ) (a).real * (x).imag - ( float  ) (a).imag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}
#define bli_zcinvscals( a, x ) \
{ \
	float  temp =              ( float  ) (a).real * (a).real + ( float  ) (a).imag * (a).imag; \
	float  xr   = ( float  ) ( ( float  ) (a).real * (x).real + ( float  ) (a).imag * (x).imag ) / temp; \
	float  xi   = ( float  ) ( ( float  ) (a).real * (x).imag - ( float  ) (a).imag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}

#define bli_szinvscals( a, x ) \
{ \
	(x).real /= ( double ) (a); \
	(x).imag /= ( double ) (a); \
}
#define bli_dzinvscals( a, x ) \
{ \
	(x).real /= ( double ) (a); \
	(x).imag /= ( double ) (a); \
}
#define bli_czinvscals( a, x ) \
{ \
	double temp =              ( double ) (a).real * (a).real + ( double ) (a).imag * (a).imag; \
	double xr   = ( double ) ( ( double ) (a).real * (x).real + ( double ) (a).imag * (x).imag ) / temp; \
	double xi   = ( double ) ( ( double ) (a).real * (x).imag - ( double ) (a).imag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}
#define bli_zzinvscals( a, x ) \
{ \
	double temp =              ( double ) (a).real * (a).real + ( double ) (a).imag * (a).imag; \
	double xr   = ( double ) ( ( double ) (a).real * (x).real + ( double ) (a).imag * (x).imag ) / temp; \
	double xi   = ( double ) ( ( double ) (a).real * (x).imag - ( double ) (a).imag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}


#define bli_sinvscals( a, x ) \
{ \
	bli_ssinvscals( a, x ); \
}
#define bli_dinvscals( a, x ) \
{ \
	bli_ddinvscals( a, x ); \
}
#define bli_cinvscals( a, x ) \
{ \
	bli_ccinvscals( a, x ); \
}
#define bli_zinvscals( a, x ) \
{ \
	bli_zzinvscals( a, x ); \
}


#endif
