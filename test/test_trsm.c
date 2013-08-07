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

//           side   uplo   trans  diag   m     n     alpha    a        lda   b        ldb
//void dtrsm_( char*, char*, char*, char*, int*, int*, double*, double*, int*, double*, int* );

//#define PRINT

int main( int argc, char** argv )
{
	obj_t a, b, c;
	obj_t c_save;
	obj_t alpha, beta;
	dim_t m, n;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input, n_input;
	num_t dt_a, dt_b, dt_c;
	num_t dt_alpha, dt_beta;
	int   r, n_repeats;
	side_t side;

#if 0
	blksz_t* mr;
	blksz_t* nr;
	blksz_t* kr;
	blksz_t* mc;
	blksz_t* nc;
	blksz_t* kc;
	blksz_t* ni;

	scalm_t* scalm_cntl;
	packm_t* packm_cntl_a;
	packm_t* packm_cntl_b;

	gemm_t*  gemm_cntl_bp_ke;
	trsm_t*  trsm_cntl_bp_ke;
	trsm_t*  trsm_cntl_op_bp;
	trsm_t*  trsm_cntl_mm_op;
	trsm_t*  trsm_cntl_vl_mm;
#endif

	double dtime;
	double dtime_save;
	double gflops;

	bli_init();

	n_repeats = 3;

#ifndef PRINT
	p_begin = 40;
	p_end   = 1000;
	p_inc   = 40;

	m_input = -1;
	//n_input = -1;
	n_input = 600;
#else
	p_begin = 16;
	p_end   = 16;
	p_inc   = 1;

	m_input = 7 ;
	n_input = 7 ;
#endif

	dt_a = BLIS_DOUBLE;
	dt_b = BLIS_DOUBLE;
	dt_c = BLIS_DOUBLE;
	dt_alpha = BLIS_DOUBLE;
	dt_beta = BLIS_DOUBLE;

	//side = BLIS_LEFT;
	side = BLIS_RIGHT;

	for ( p = p_begin; p <= p_end; p += p_inc )
	{

		if ( m_input < 0 ) m = p * ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;
		if ( n_input < 0 ) n = p * ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;


		bli_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );
		bli_obj_create( dt_beta,  1, 1, 0, 0, &beta );

		if ( bli_is_left( side ) )
			bli_obj_create( dt_a, m, m, 0, 0, &a );
		else
			bli_obj_create( dt_a, n, n, 0, 0, &a );
		bli_obj_create( dt_b, m, n, 0, 0, &b );
		bli_obj_create( dt_c, m, n, 0, 0, &c );
		bli_obj_create( dt_c, m, n, 0, 0, &c_save );

		bli_obj_set_struc( BLIS_TRIANGULAR, a );
		//bli_obj_set_uplo( BLIS_UPPER, a );
		bli_obj_set_uplo( BLIS_LOWER, a );

		bli_randm( &a );
		bli_randm( &c );
		bli_randm( &b );

		bli_setsc(  (2.0/1.0), 0.0, &alpha );
		bli_setsc( -(1.0/1.0), 0.0, &beta );

#if 0
		mr = bli_blksz_obj_create( 2, 4, 2, 2 );
		kr = bli_blksz_obj_create( 1, 1, 1, 1 );
		nr = bli_blksz_obj_create( 1, 4, 1, 1 );
		mc = bli_blksz_obj_create( 128, 368, 128, 128 );
		kc = bli_blksz_obj_create( 256, 256, 256, 256 );
		nc = bli_blksz_obj_create( 512, 512, 512, 512 );
		ni = bli_blksz_obj_create(  16,  16,  16,  16 );

		scalm_cntl =
		bli_scalm_cntl_obj_create( BLIS_UNBLOCKED,
		                           BLIS_VARIANT1 );

		packm_cntl_a =
		bli_packm_cntl_obj_create( BLIS_BLOCKED,
		                           BLIS_VARIANT3,
		                           mr, // IMPORTANT: "k" dim multiple must be mr to
		                           mr, // support using ukernel for right/bottom-right
		                                  // edge cases (see macro-kernel for comments).
		                           FALSE, // scale?
		                           TRUE,  // densify?
		                           TRUE,  // invert diagonal?
		                           TRUE,  // reverse iteration if upper?
		                           FALSE, // reverse iteration if lower?
		                           BLIS_PACKED_ROW_PANELS,
		                           BLIS_BUFFER_FOR_A_BLOCK );

		packm_cntl_b =
		bli_packm_cntl_obj_create( BLIS_BLOCKED,
		                           BLIS_VARIANT2,
		                           mr, // IMPORTANT: m dim multiple here must be mr
		                           nr, // since "k" dim multiple is set to mr above.
		                           TRUE,  // scale?
		                           FALSE, // densify?
		                           FALSE, // invert diagonal?
		                           FALSE, // reverse iteration if upper?
		                           FALSE, // reverse iteration if lower?
		                           BLIS_PACKED_COL_PANELS,
		                           BLIS_BUFFER_FOR_B_PANEL );

		gemm_cntl_bp_ke =
		bli_gemm_cntl_obj_create( BLIS_UNB_OPT,
		                          BLIS_VARIANT2,
		                          NULL, NULL, NULL, NULL,
		                          NULL, NULL, NULL, NULL );

		trsm_cntl_bp_ke =
		bli_trsm_cntl_obj_create( BLIS_UNB_OPT,
		                          BLIS_VARIANT2,
		                          //BLIS_VARIANT3,
		                          NULL, NULL, NULL, NULL, NULL,
		                          NULL, NULL, NULL, NULL );

		trsm_cntl_op_bp =
		bli_trsm_cntl_obj_create( BLIS_BLOCKED,
		                          //BLIS_VARIANT4,
		                          BLIS_VARIANT1,
		                          mc,
		                          ni,
		                          NULL,
		                          packm_cntl_a,
		                          packm_cntl_b,
		                          NULL,
		                          trsm_cntl_bp_ke,
		                          gemm_cntl_bp_ke,
		                          NULL );

		trsm_cntl_mm_op =
		bli_trsm_cntl_obj_create( BLIS_BLOCKED,
		                          BLIS_VARIANT3,
		                          kc,
		                          NULL,
		                          NULL, //scalm_cntl,
		                          NULL,
		                          NULL,
		                          NULL,
		                          trsm_cntl_op_bp,
		                          NULL,
		                          NULL );

		trsm_cntl_vl_mm =
		bli_trsm_cntl_obj_create( BLIS_BLOCKED,
		                          BLIS_VARIANT2,
		                          nc,
		                          NULL,
		                          NULL,
		                          NULL,
		                          NULL,
		                          NULL,
		                          trsm_cntl_mm_op,
		                          NULL,
		                          NULL );
#endif


		bli_copym( &c, &c_save );
	
		dtime_save = 1.0e9;

		for ( r = 0; r < n_repeats; ++r )
		{
			bli_copym( &c_save, &c );

			dtime = bli_clock();


#ifdef PRINT
			bli_invertd( &a );
			bli_printm( "a", &a, "%4.1f", "" );
			bli_invertd( &a );
			bli_printm( "c", &c, "%4.1f", "" );
#endif

#ifdef BLIS
			//bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

			//bli_obj_set_diag( BLIS_UNIT_DIAG, a );
			//bli_obj_set_conjtrans( BLIS_TRANSPOSE, a );

			bli_trsm( side,
			          &alpha,
			          &a,
			          &c );

#else

			f77_char side   = 'R';
			f77_char uplo   = 'L';
			f77_char transa = 'N';
			f77_char diag   = 'N';
			f77_int  mm     = bli_obj_length( c );
			f77_int  nn     = bli_obj_width( c );
			f77_int  lda    = bli_obj_col_stride( a );
			f77_int  ldc    = bli_obj_col_stride( c );
			double*  alphap = bli_obj_buffer( alpha );
			double*  ap     = bli_obj_buffer( a );
			double*  cp     = bli_obj_buffer( c );

			dtrsm_( &side,
			        &uplo,
			        &transa,
			        &diag,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        cp, &ldc );
#endif

#ifdef PRINT
			bli_printm( "c after", &c, "%4.1f", "" );
			exit(1);
#endif


			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}

		if ( bli_is_left( side ) )
			gflops = ( 1.0 * m * m * n ) / ( dtime_save * 1.0e9 );
		else
			gflops = ( 1.0 * m * n * n ) / ( dtime_save * 1.0e9 );

#ifdef BLIS
		printf( "data_trsm_blis" );
#else
		printf( "data_trsm_%s", BLAS );
#endif
		printf( "( %2ld, 1:4 ) = [ %4lu %4lu  %10.3e  %6.3f ];\n",
		        (p - p_begin + 1)/p_inc + 1, m, n, dtime_save, gflops );

#if 0
		bli_blksz_obj_free( mr );
		bli_blksz_obj_free( nr );
		bli_blksz_obj_free( kr );
		bli_blksz_obj_free( mc );
		bli_blksz_obj_free( nc );
		bli_blksz_obj_free( kc );
		bli_blksz_obj_free( ni );

		bli_cntl_obj_free( scalm_cntl );
		bli_cntl_obj_free( packm_cntl_a );
		bli_cntl_obj_free( packm_cntl_b );
		bli_cntl_obj_free( trsm_cntl_bp_ke );
		bli_cntl_obj_free( trsm_cntl_op_bp );
		bli_cntl_obj_free( trsm_cntl_mm_op );
		bli_cntl_obj_free( trsm_cntl_vl_mm );
#endif

		bli_obj_free( &alpha );
		bli_obj_free( &beta );

		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );
		bli_obj_free( &c_save );
	}

	bli_finalize();

	return 0;
}

