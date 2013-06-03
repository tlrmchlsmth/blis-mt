/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

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

//           transa transb m     n     k     alpha    a        lda   b        ldb   beta     c        ldc
#define PRINT 1

void dgemm_reference(double alpha, obj_t A, obj_t B, obj_t C)
{
    double * a = (double *) A.buffer;
    double * b = (double *) B.buffer;
    double * c = (double *) C.buffer;
    
    for(int i = 0; i < C.m; i++)
    {
        for(int j = 0; j < C.n; j++)
        {
            for(int k = 0; k < A.n; k++)
            {
                c[i*C.rs + j*C.cs] = alpha * a[i*A.rs + k*A.cs] *b[k*B.rs + j*B.cs]  + c[i*C.rs + j*C.cs];
            }
        }
    }
}

void dgemm_( char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int* );

int main( int argc, char** argv )
{
	obj_t a, b, c;
	obj_t a_pack, b_pack;
	obj_t c_save;
	obj_t alpha, beta;
	dim_t m, n, k;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input, n_input, k_input;
	num_t dt_a, dt_b, dt_c;
	num_t dt_alpha, dt_beta;
	int   r, n_repeats;

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
	gemm_t*  gemm_cntl_op_bp;
	gemm_t*  gemm_cntl_mm_op;
	gemm_t*  gemm_cntl_vl_mm;

	double dtime;
	double dtime_save;
	double gflops;

	bli_init();

    int thr_l2, thr_l3, thr_l4, thr_l5, thr_pack_a, thr_pack_b;
    if(argc != 5){
        thr_l2 = 1;
        thr_l3 = 1;
        thr_l4 = 1;
        thr_l5 = 1;
    }
    else{
        sscanf( argv[1], "%d", &thr_l2 );
        sscanf( argv[2], "%d", &thr_l3 );
        sscanf( argv[3], "%d", &thr_l4 );
        sscanf( argv[4], "%d", &thr_l5 );
    }
    thr_pack_a = thr_l2;
    thr_pack_b = thr_l3 * thr_l2;


	n_repeats = 3;

#ifndef PRINT
	p_begin = 40;
	p_end   = 2000;
	p_inc   = 40;

	m_input = -1;
	//m_input = 384;
	n_input = -1;
	//k_input = -1;
	k_input = 200;
#else
	p_begin = 24;
	p_end   = 24;
	p_inc   = 1;

	//m_input = 10;
	//k_input = 10;
	//n_input = 10;
    m_input = -1;
    n_input = -1;
    k_input = -1;
#endif

	dt_a = BLIS_DOUBLE;
	dt_b = BLIS_DOUBLE;
	dt_c = BLIS_DOUBLE;
	dt_alpha = BLIS_DOUBLE;
	dt_beta = BLIS_DOUBLE;

	for ( p = p_begin; p <= p_end; p += p_inc )
	{

		if ( m_input < 0 ) m = p * ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;
		if ( n_input < 0 ) n = p * ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;
		if ( k_input < 0 ) k = p * ( dim_t )abs(k_input);
		else               k =     ( dim_t )    k_input;


		bli_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );
		bli_obj_create( dt_beta,  1, 1, 0, 0, &beta );

		bli_obj_create( dt_a, m, k, 0, 0, &a );
		bli_obj_create( dt_b, k, n, 0, 0, &b );
		bli_obj_create( dt_c, m, n, 0, 0, &c );
		bli_obj_create( dt_c, m, n, 0, 0, &c_save );

		bli_randm( &a );
		bli_randm( &b );
		bli_randm( &c );


		bli_setsc(  (2.0/1.0),0.0, &alpha );
		bli_setsc(  (1.0/1.0),0.0, &beta );
		mr = bli_blksz_obj_create( 2, 4, 2, 2 );
		kr = bli_blksz_obj_create( 1, 1, 1, 1 );
		nr = bli_blksz_obj_create( 1, 4, 1, 1 );
		mc = bli_blksz_obj_create( 128, 128, 128, 128 );
		kc = bli_blksz_obj_create( 256, 256, 256, 256 );
		nc = bli_blksz_obj_create( 512, 512, 512, 512 );
		ni = bli_blksz_obj_create(  16,  16,  16,  16 );

		bli_obj_init_pack( &a_pack );
		bli_obj_init_pack( &b_pack );


		bli_copym( &c, &c_save );
	
		dtime_save = 1.0e9;

		for ( r = 0; r < n_repeats; ++r )
		{
			bli_copym( &c_save, &c );


			dtime = bli_clock();


#ifdef PRINT
			bli_printm( "a", &a, "%4.2f", "" );
			bli_printm( "b", &b, "%4.2f", "" );
			bli_printm( "c", &c, "%4.2f", "" );
#endif

			//bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

			bli_gemm( &alpha,
			              &a,
			              &b,
			              &beta,
			              &c );

             dgemm_reference(2.0, a, b, c_save);
#ifdef PRINT
		    bli_setsc(  (-1.0/1.0), 0.0, &beta );
			bli_printm( "c after", &c, "%4.2f", "" );
            bli_printm( "c refer", &c_save, "%4.2f", "" );
            bli_axpym( &beta, &c_save, &c);
			bli_printm( "c diff", &c, "%4.4f", "" );
			exit(1);
#endif


			dtime = bli_clock() - dtime;

			dtime_save = bli_min( dtime, dtime_save );

		}

		gflops = ( 2.0 * m * k * n ) / ( dtime_save * 1.0e9 );

#ifdef BLIS
		printf( "data_gemm_blis" );
#else
		printf( "data_gemm_%s", BLAS );
#endif
		printf( "( %2ld, 1:5 ) = [ %4lu %4lu %4lu  %10.3e  %6.3f ];\n",
		        (p - p_begin + 1)/p_inc + 1, m, k, n, dtime_save, gflops );

		bli_obj_release_pack( &a_pack );
		bli_obj_release_pack( &b_pack );

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
		bli_cntl_obj_free( gemm_cntl_bp_ke );
		bli_cntl_obj_free( gemm_cntl_op_bp );
		bli_cntl_obj_free( gemm_cntl_mm_op );
		bli_cntl_obj_free( gemm_cntl_vl_mm );

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

