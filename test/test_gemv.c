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

#include <unistd.h>
#include "blis.h"

//           transa m     n     alpha    a        lda   x        incx  beta     y        incy
<<<<<<< HEAD
//void dgemv_( char*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int* );

//#define PRINT
=======
void dgemv_( char*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int* );
>>>>>>> 0c1c78278bbd9c281bcbe933cc2f3bdb3bd74ef1

int main( int argc, char** argv )
{
	obj_t a, x, y;
	obj_t a_tl, x_t, y_t;
	obj_t y_save;
	obj_t alpha, beta;
	dim_t m, n;
	dim_t m_tl, n_tl;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input, n_input;
	num_t dt_a, dt_x, dt_y;
	num_t dt_alpha, dt_beta;
	int   r, n_repeats;

	double dtime;
	double dtime_save;
	double gflops;

	bli_init();

	n_repeats = 3;

#ifndef PRINT
<<<<<<< HEAD
	p_begin = 40;
	p_end   = 2000;
	p_inc   = 40;
=======
	p_begin = 64;
	p_end   = 4096;
	p_inc   = 64;
>>>>>>> 0c1c78278bbd9c281bcbe933cc2f3bdb3bd74ef1

	m_input = -1;
	n_input = -1;
#else
	p_begin = 16;
	p_end   = 16;
	p_inc   = 1;

	m_input = 15;
	n_input = 15;
#endif

	dt_a = BLIS_DOUBLE;
	dt_x = BLIS_DOUBLE;
	dt_y = BLIS_DOUBLE;
	dt_alpha = BLIS_DOUBLE;
	dt_beta = BLIS_DOUBLE;

	for ( p = p_begin; p <= p_end; p += p_inc )
	{

		if ( m_input < 0 ) m = p * ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;
		if ( n_input < 0 ) n = p * ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;


		bli_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );
		bli_obj_create( dt_beta,  1, 1, 0, 0, &beta );

		bli_obj_create( dt_a, m, n, 0, 0, &a );
		bli_obj_create( dt_x, n, 1, 0, 0, &x );
		bli_obj_create( dt_y, m, 1, 0, 0, &y );
		bli_obj_create( dt_y, m, 1, 0, 0, &y_save );

		bli_randm( &a );
		bli_randm( &x );
		bli_randm( &y );


		bli_setsc(  (2.0/1.0), 0.0, &alpha );
<<<<<<< HEAD
		bli_setsc( -(1.0/1.0), 0.0, &beta );
=======
		bli_setsc(  (1.0/1.0), 0.0, &beta );
>>>>>>> 0c1c78278bbd9c281bcbe933cc2f3bdb3bd74ef1

#if 0
		m_tl = 200;
		n_tl = 200;

		m_tl = bli_min( m_tl, bli_obj_length( a ) );
		n_tl = bli_min( n_tl, bli_obj_width( a ) );

		bli_acquire_mpart_tl2br( BLIS_SUBPART11, 0, m_tl, &a, &a_tl );
		bli_acquire_mpart_t2b( BLIS_SUBPART1, 0, n_tl, &x, &x_t );
		bli_acquire_mpart_t2b( BLIS_SUBPART1, 0, m_tl, &y, &y_t );
#else
		m_tl = m;
		n_tl = n;

		a_tl = a;
		x_t  = x;
		y_t  = y;
#endif


		bli_copym( &y, &y_save );
	
		dtime_save = 1.0e9;

		for ( r = 0; r < n_repeats; ++r )
		{
			bli_copym( &y_save, &y );


			dtime = bli_clock();

#ifdef PRINT
			bli_printm( "a", &a, "%4.1f", "" );
			bli_printm( "x", &x, "%4.1f", "" );
			bli_printm( "y", &y, "%4.1f", "" );
#endif

#ifdef BLIS
<<<<<<< HEAD
			//bli_obj_set_onlytrans( BLIS_TRANSPOSE, a_tl );
=======
			//bli_obj_set_trans( BLIS_TRANSPOSE, a_tl );
>>>>>>> 0c1c78278bbd9c281bcbe933cc2f3bdb3bd74ef1

			bli_gemv( &alpha,
			          &a_tl,
			          &x_t,
			          &beta,
			          &y_t );

#else

<<<<<<< HEAD
			f77_char transa = 'N';
			f77_int  mm     = bli_obj_length( a_tl );
			f77_int  nn     = bli_obj_width( a_tl );
			f77_int  lda    = bli_obj_col_stride( a_tl );
			f77_int  incx   = bli_obj_vector_inc( x_t );
			f77_int  incy   = bli_obj_vector_inc( y_t );
			double*  alphap = bli_obj_buffer( alpha );
			double*  ap     = bli_obj_buffer( a_tl );
			double*  xp     = bli_obj_buffer( x_t );
			double*  betap  = bli_obj_buffer( beta );
			double*  yp     = bli_obj_buffer( y_t );
=======
			char    transa = 'N';
			int     mm     = bli_obj_length( a_tl );
			int     nn     = bli_obj_width( a_tl );
			int     lda    = bli_obj_col_stride( a_tl );
			int     incx   = bli_obj_vector_inc( x_t );
			int     incy   = bli_obj_vector_inc( y_t );
			double* alphap = bli_obj_buffer( alpha );
			double* ap     = bli_obj_buffer( a_tl );
			double* xp     = bli_obj_buffer( x_t );
			double* betap  = bli_obj_buffer( beta );
			double* yp     = bli_obj_buffer( y_t );
>>>>>>> 0c1c78278bbd9c281bcbe933cc2f3bdb3bd74ef1

			dgemv_( &transa,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        xp, &incx,
			        betap,
			        yp, &incy );
#endif

#ifdef PRINT
			bli_printm( "y after", &y, "%4.1f", "" );
			exit(1);
#endif


			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}

		//gflops = ( 2.0 * m * n ) / ( dtime_save * 1.0e9 );
		gflops = ( 2.0 * m_tl * n_tl ) / ( dtime_save * 1.0e9 );

#ifdef BLIS
		printf( "data_gemv_blis" );
#else
		printf( "data_gemv_%s", BLAS );
#endif
		printf( "( %2ld, 1:4 ) = [ %4lu %4lu  %10.3e  %6.3f ];\n",
		        (p - p_begin + 1)/p_inc + 1, m, n, dtime_save, gflops );

		bli_obj_free( &alpha );
		bli_obj_free( &beta );

		bli_obj_free( &a );
		bli_obj_free( &x );
		bli_obj_free( &y );
		bli_obj_free( &y_save );
	}

	bli_finalize();

	return 0;
}

