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

#ifndef BLIS_DOTJS_H
#define BLIS_DOTJS_H

// dotjs

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.
// - The third char encodes the type of rho.
// - x is used in conjugated form.

// -- (xyr) = (ss?) ------------------------------------------------------------

#define bli_sssdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bli_ssddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bli_sscdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_sszdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (sd?) ------------------------------------------------------------

#define bli_sdsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bli_sdddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bli_sdcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_sdzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (sc?) ------------------------------------------------------------

#define bli_scsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bli_scddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bli_sccdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_sczdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (sz?) ------------------------------------------------------------

#define bli_szsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bli_szddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bli_szcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_szzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (ds?) ------------------------------------------------------------

#define bli_dssdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bli_dsddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bli_dscdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_dszdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (dd?) ------------------------------------------------------------

#define bli_ddsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y); \
}
#define bli_ddddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y); \
}
#define bli_ddcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y); \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_ddzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y); \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (dc?) ------------------------------------------------------------

#define bli_dcsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bli_dcddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bli_dccdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_dczdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (dz?) ------------------------------------------------------------

#define bli_dzsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x) * ( float  ) (y).real; \
}
#define bli_dzddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x) * ( double ) (y).real; \
}
#define bli_dzcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x) * ( float  ) (y).real; \
	/* (a).imag += ( float  ) 0.0; */ \
}
#define bli_dzzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x) * ( double ) (y).real; \
	/* (a).imag += ( double ) 0.0; */ \
}

// -- (xyr) = (cs?) ------------------------------------------------------------

#define bli_cssdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bli_csddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bli_cscdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  )-(x).imag * ( float  ) (y); \
}
#define bli_cszdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double )-(x).imag * ( double ) (y); \
}

// -- (xyr) = (cd?) ------------------------------------------------------------

#define bli_cdsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bli_cdddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bli_cdcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  )-(x).imag * ( float  ) (y); \
}
#define bli_cdzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double )-(x).imag * ( double ) (y); \
}

// -- (xyr) = (cc?) ------------------------------------------------------------

#define bli_ccsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bli_ccddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bli_cccdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real + ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag - ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bli_cczdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real + ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag - ( double ) (x).imag * ( double ) (y).real; \
}

// -- (xyr) = (cz?) ------------------------------------------------------------

#define bli_czsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bli_czddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bli_czcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real + ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag - ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bli_czzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real + ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag - ( double ) (x).imag * ( double ) (y).real; \
}

// -- (xyr) = (zs?) ------------------------------------------------------------

#define bli_zssdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bli_zsddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bli_zscdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  ) (x).imag * ( float  ) (y); \
}
#define bli_zszdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double ) (x).imag * ( double ) (y); \
}

// -- (xyr) = (zd?) ------------------------------------------------------------

#define bli_zdsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y); \
}
#define bli_zdddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y); \
}
#define bli_zdcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y); \
	(a).imag += ( float  ) (x).imag * ( float  ) (y); \
}
#define bli_zdzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y); \
	(a).imag += ( double ) (x).imag * ( double ) (y); \
}

// -- (xyr) = (zc?) ------------------------------------------------------------

#define bli_zcsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bli_zcddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bli_zccdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real + ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag - ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bli_zczdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real + ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag - ( double ) (x).imag * ( double ) (y).real; \
}

// -- (xyr) = (zz?) ------------------------------------------------------------

#define bli_zzsdotjs( x, y, a ) \
{ \
	(a)      += ( float  ) (x).real * ( float  ) (y).real - ( float  ) (x).imag * ( float  ) (y).imag; \
}
#define bli_zzddotjs( x, y, a ) \
{ \
	(a)      += ( double ) (x).real * ( double ) (y).real - ( double ) (x).imag * ( double ) (y).imag; \
}
#define bli_zzcdotjs( x, y, a ) \
{ \
	(a).real += ( float  ) (x).real * ( float  ) (y).real + ( float  ) (x).imag * ( float  ) (y).imag; \
	(a).imag += ( float  ) (x).real * ( float  ) (y).imag - ( float  ) (x).imag * ( float  ) (y).real; \
}
#define bli_zzzdotjs( x, y, a ) \
{ \
	(a).real += ( double ) (x).real * ( double ) (y).real + ( double ) (x).imag * ( double ) (y).imag; \
	(a).imag += ( double ) (x).real * ( double ) (y).imag - ( double ) (x).imag * ( double ) (y).real; \
}



#define bli_sdotjs( x, y, a ) \
{ \
	bli_sssdotjs( x, y, a ); \
}
#define bli_ddotjs( x, y, a ) \
{ \
	bli_ddddotjs( x, y, a ); \
}
#define bli_cdotjs( x, y, a ) \
{ \
	bli_cccdotjs( x, y, a ); \
}
#define bli_zdotjs( x, y, a ) \
{ \
	bli_zzzdotjs( x, y, a ); \
}


#endif
