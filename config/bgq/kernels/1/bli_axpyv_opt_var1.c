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

#include "blis.h"

#define FUNCPTR_T axpyv_fp

typedef void (*FUNCPTR_T)(
                           conj_t conjx,
                           dim_t  n,
                           void*  alpha,
                           void*  x, inc_t incx,
                           void*  y, inc_t incy
                         );

// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY3_ALL(ftypes,axpyv_opt_var1);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY3_EXT(ftypes,axpyv_opt_var1);
#else
static FUNCPTR_T GENARRAY3_MIN(ftypes,axpyv_opt_var1);
#endif
#endif


void bli_axpyv_opt_var1( obj_t*  alpha,
                         obj_t*  x,
                         obj_t*  y )
{
    num_t     dt_x      = bli_obj_datatype( *x );
    num_t     dt_y      = bli_obj_datatype( *y );

    conj_t    conjx     = bli_obj_conj_status( *x );
    dim_t     n         = bli_obj_vector_dim( *x );

    inc_t     inc_x     = bli_obj_vector_inc( *x );
    void*     buf_x     = bli_obj_buffer_at_off( *x );

    inc_t     inc_y     = bli_obj_vector_inc( *y );
    void*     buf_y     = bli_obj_buffer_at_off( *y );

    num_t     dt_alpha;
    void*     buf_alpha;

    FUNCPTR_T f;

    // If alpha is a scalar constant, use dt_x to extract the address of the
    // corresponding constant value; otherwise, use the datatype encoded
    // within the alpha object and extract the buffer at the alpha offset.
    bli_set_scalar_dt_buffer( alpha, dt_x, dt_alpha, buf_alpha );

    // Index into the type combination array to extract the correct
    // function pointer.
    f = ftypes[dt_alpha][dt_x][dt_y];

    // Invoke the function.
    f( conjx,
       n,
       buf_alpha,
       buf_x, inc_x,
       buf_y, inc_y );
}


#undef  GENTFUNC3
#define GENTFUNC3( ctype_a, ctype_x, ctype_y, cha, chx, chy, opname, varname ) \
\
void PASTEMAC3(cha,chx,chy,varname)( \
                                     conj_t conjx, \
                                     dim_t  n, \
                                     void*  alpha, \
                                     void*  x, inc_t incx, \
                                     void*  y, inc_t incy \
                                   ) \
{ \
    ctype_a* alpha_cast = alpha; \
    ctype_x* x_cast     = x; \
    ctype_y* y_cast     = y; \
    ctype_x* chi1; \
    ctype_y* psi1; \
    dim_t    i; \
\
    if ( bli_zero_dim1( n ) ) return; \
\
    chi1 = x_cast; \
    psi1 = y_cast; \
\
    if ( bli_is_conj( conjx ) ) \
    { \
        for ( i = 0; i < n; ++i ) \
        { \
            PASTEMAC3(cha,chx,chy,axpyjs)( *alpha_cast, *chi1, *psi1 ); \
\
            chi1 += incx; \
            psi1 += incy; \
        } \
    } \
    else \
    { \
        for ( i = 0; i < n; ++i ) \
        { \
            PASTEMAC3(cha,chx,chy,axpys)( *alpha_cast, *chi1, *psi1 ); \
\
            chi1 += incx; \
            psi1 += incy; \
        } \
    } \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
//INSERT_GENTFUNC3_BASIC( axpyv, axpyv_opt_var1 )
GENTFUNC3( float,    float,    float,    s, s, s, axpyv, axpyv_opt_var1 )
//GENTFUNC3( double,   double,   double,   d, d, d, axpyv, axpyv_opt_var1 )
GENTFUNC3( scomplex, scomplex, scomplex, c, c, c, axpyv, axpyv_opt_var1 )
GENTFUNC3( dcomplex, dcomplex, dcomplex, z, z, z, axpyv, axpyv_opt_var1 )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3_MIX_D( axpyv, axpyv_opt_var1 )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3_MIX_P( axpyv, axpyv_opt_var1 )
#endif

void bli_dddaxpyv_opt_var1( 
                            conj_t conjx,
                            dim_t  n,
                            void*  alpha_in,
                            void*  x_in, inc_t incx,
                            void*  y_in, inc_t incy
                          )
{
    double*  restrict alpha = alpha_in;
    double*  restrict x = x_in;
    double*  restrict y = y_in;

    if ( bli_zero_dim1( n ) ) return;

    // If there is anything that would interfere with our use of aligned
    // vector loads/stores, call the reference implementation.
    bool_t use_ref = FALSE;
    if ( incx != 1 || incy != 1 || bli_is_unaligned_to( x, 32 ) || bli_is_unaligned_to( y, 32 ) ) {
        use_ref = TRUE;
    }
    // Call the reference implementation if needed.
    if ( use_ref == TRUE ) {
        bli_dddaxpyv_unb_var1( conjx, n, alpha, x, incx, y, incy );
        return;
    }

    dim_t n_run       = n / 4;
    dim_t n_left      = n % 4;

    vector4double xv, yv, zv;
    vector4double alphav = vec_lds( 0 * sizeof(double), alpha );

    #ifndef BLIS_DISABLE_THREADING
    _Pragma("omp parallel for")
    #endif
    for ( dim_t i = 0; i < n_run; i++ )
    {
        xv = vec_lda( 0 * sizeof(double), &x[i*4] );
        yv = vec_lda( 0 * sizeof(double), &y[i*4] );
        zv = vec_madd( alphav, xv, yv );
        vec_sta( zv, 0 * sizeof(double), &y[i*4] );   
    }
    for ( dim_t i = 0; i < n_left; i++ )
    {
        y[4*n_run + i] += *alpha * x[4*n_run + i];
    }
}
