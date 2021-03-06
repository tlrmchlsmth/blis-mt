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

#ifndef BLIS_XPBYS_MXN_UPLO_H
#define BLIS_XPBYS_MXN_UPLO_H

// xpbys_mxn_u

#define bli_sssxpbys_mxn_u( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{ \
	dim_t i, j; \
\
	for ( j = 0; j < n; ++j ) \
	{ \
		for ( i = 0; i < m; ++i ) \
		{ \
			if ( (doff_t)j - (doff_t)i >= diagoff ) \
			{ \
				bli_sssxpbys( *(x + i*rs_x + j*cs_x), \
				              *(beta), \
				              *(y + i*rs_y + j*cs_y) ); \
			} \
		} \
	} \
}

#define bli_dddxpbys_mxn_u( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{ \
	dim_t i, j; \
\
	for ( j = 0; j < n; ++j ) \
	{ \
		for ( i = 0; i < m; ++i ) \
		{ \
			if ( (doff_t)j - (doff_t)i >= diagoff ) \
			{ \
				bli_dddxpbys( *(x + i*rs_x + j*cs_x), \
				              *(beta), \
				              *(y + i*rs_y + j*cs_y) ); \
			} \
		} \
	} \
}

#define bli_cccxpbys_mxn_u( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{ \
	dim_t i, j; \
\
	for ( j = 0; j < n; ++j ) \
	{ \
		for ( i = 0; i < m; ++i ) \
		{ \
			if ( (doff_t)j - (doff_t)i >= diagoff ) \
			{ \
				bli_cccxpbys( *(x + i*rs_x + j*cs_x), \
				              *(beta), \
				              *(y + i*rs_y + j*cs_y) ); \
			} \
		} \
	} \
}

#define bli_zzzxpbys_mxn_u( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{ \
	dim_t i, j; \
\
	for ( j = 0; j < n; ++j ) \
	{ \
		for ( i = 0; i < m; ++i ) \
		{ \
			if ( (doff_t)j - (doff_t)i >= diagoff ) \
			{ \
				bli_zzzxpbys( *(x + i*rs_x + j*cs_x), \
				              *(beta), \
				              *(y + i*rs_y + j*cs_y) ); \
			} \
		} \
	} \
}

// xpbys_mxn_l

#define bli_sssxpbys_mxn_l( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{ \
	dim_t i, j; \
\
	for ( j = 0; j < n; ++j ) \
	{ \
		for ( i = 0; i < m; ++i ) \
		{ \
			if ( (doff_t)j - (doff_t)i <= diagoff ) \
			{ \
				bli_sssxpbys( *(x + i*rs_x + j*cs_x), \
				              *(beta), \
				              *(y + i*rs_y + j*cs_y) ); \
			} \
		} \
	} \
}

#define bli_dddxpbys_mxn_l( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{ \
	dim_t i, j; \
\
	for ( j = 0; j < n; ++j ) \
	{ \
		for ( i = 0; i < m; ++i ) \
		{ \
			if ( (doff_t)j - (doff_t)i <= diagoff ) \
			{ \
				bli_dddxpbys( *(x + i*rs_x + j*cs_x), \
				              *(beta), \
				              *(y + i*rs_y + j*cs_y) ); \
			} \
		} \
	} \
}

#define bli_cccxpbys_mxn_l( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{ \
	dim_t i, j; \
\
	for ( j = 0; j < n; ++j ) \
	{ \
		for ( i = 0; i < m; ++i ) \
		{ \
			if ( (doff_t)j - (doff_t)i <= diagoff ) \
			{ \
				bli_cccxpbys( *(x + i*rs_x + j*cs_x), \
				              *(beta), \
				              *(y + i*rs_y + j*cs_y) ); \
			} \
		} \
	} \
}

#define bli_zzzxpbys_mxn_l( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{ \
	dim_t i, j; \
\
	for ( j = 0; j < n; ++j ) \
	{ \
		for ( i = 0; i < m; ++i ) \
		{ \
			if ( (doff_t)j - (doff_t)i <= diagoff ) \
			{ \
				bli_zzzxpbys( *(x + i*rs_x + j*cs_x), \
				              *(beta), \
				              *(y + i*rs_y + j*cs_y) ); \
			} \
		} \
	} \
}


#define bli_sxpbys_mxn_u( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{\
	bli_sssxpbys_mxn_u( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ); \
}
#define bli_dxpbys_mxn_u( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{\
	bli_dddxpbys_mxn_u( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ); \
}
#define bli_cxpbys_mxn_u( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{\
	bli_cccxpbys_mxn_u( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ); \
}
#define bli_zxpbys_mxn_u( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{\
	bli_zzzxpbys_mxn_u( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ); \
}
#define bli_sxpbys_mxn_l( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{\
	bli_sssxpbys_mxn_l( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ); \
}
#define bli_dxpbys_mxn_l( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{\
	bli_dddxpbys_mxn_l( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ); \
}
#define bli_cxpbys_mxn_l( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{\
	bli_cccxpbys_mxn_l( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ); \
}
#define bli_zxpbys_mxn_l( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{\
	bli_zzzxpbys_mxn_l( diagoff, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ); \
}

#endif
