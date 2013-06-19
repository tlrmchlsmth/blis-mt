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

#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <sys/time.h>

static double gtod_ref_time_sec = 0.0;

double bli_clock()
{
    double         the_time, norm_sec;
    struct timeval tv; 

    gettimeofday( &tv, NULL );

    if ( gtod_ref_time_sec == 0.0 )
        gtod_ref_time_sec = ( double ) tv.tv_sec;

    norm_sec = ( double ) tv.tv_sec - gtod_ref_time_sec;

    the_time = norm_sec + tv.tv_usec * 1.0e-6;

    return the_time;
}

double bli_clock_min_diff( double time_min, double time_start )
{
    double time_min_prev;
    double time_diff;

    // Save the old value.
    time_min_prev = time_min;

    time_diff = bli_clock() - time_start;

    time_min = time_min < time_diff ? time_min : time_diff;

    // Assume that anything:
    // - under or equal to zero,
    // - over an hour, or
    // - under a nanosecond
    // is actually garbled due to the clocks being taken too closely together.
    if      ( time_min <= 0.0    ) time_min = time_min_prev;
    else if ( time_min >  3600.0 ) time_min = time_min_prev;
    else if ( time_min <  1.0e-9 ) time_min = time_min_prev;

    return time_min;
}

//           transa transb m     n     k     alpha    a        lda   b        ldb   beta     c        ldc
void dgemm_( char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int* );

//#define PRINT
void fillrand(double * a, int n)
{   
    int max = 5;
    int min = -5;
    for(int i = 0; i < n; i++)
    {
        a[i] = ((double) rand() / (RAND_MAX+1)) * (max-min+1) + min;
    }
}

int main( int argc, char** argv )
{
    double* a;
    double* b;
    double* c;
    double* c_save;
    double alpha, beta;
    int m, n, k; 
    int p;
	int p_begin, p_end, p_inc;
	int   m_input, n_input, k_input;
	int   r, n_repeats;

	double dtime;
	double dtime_save;
	double gflops;

    int world_size, world_rank, provided;
    MPI_Init_thread( NULL, NULL, MPI_THREAD_FUNNELED, &provided );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
	
    n_repeats = 3;

	p_begin = 40;
	p_end   = 2000;
	p_inc   = 40;

	m_input = -1;
	n_input = -1;
	k_input = -1;

	for ( p = p_begin; p <= p_end; p += p_inc )
	{

		if ( m_input < 0 ) m = p * ( int )abs(m_input);
		else               m =     ( int )    m_input;
		if ( n_input < 0 ) n = p * ( int )abs(n_input);
		else               n =     ( int )    n_input;
		if ( k_input < 0 ) k = p * ( int )abs(k_input);
		else               k =     ( int )    k_input;

        a = (double*)malloc( m * k * sizeof(double));
        b = (double*)malloc( k * n * sizeof(double));
        c = (double*)malloc( m * n * sizeof(double));
        c_save = (double*)malloc( m * n * sizeof(double));
    
        fillrand(a, m*k);
        fillrand(b, m*k);
        fillrand(c, m*k);
        memcpy( c_save, c, m*n*sizeof(double) );

        alpha = 1.0;
        beta = 1.0;

		dtime_save = 1.0e9;

		for ( r = 0; r < n_repeats; ++r )
		{

            memcpy( c_save, c, m*n*sizeof(double) );

			dtime = bli_clock();



			char    transa = 'N';
			char    transb = 'N';
			int     lda    = m;
			int     ldb    = k;
			int     ldc    = m;

			dgemm_( &transa,
			        &transb,
			        &m,
			        &n,
			        &k,
			        &alpha,
			        a, &lda,
			        b, &ldb,
			        &beta,
			        c, &ldc );
            

			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}
		
        gflops = ( 2.0 * m * k * n ) / ( dtime_save * 1.0e9 );

        if(world_rank == 0)
        {
            printf( "data_gemm_essl" );
            printf( "( %2ld, 1:5 ) = [ %4lu %4lu %4lu  %10.3e  %6.3f ];\n",
                    (p - p_begin + 1)/p_inc + 1, m, k, n, dtime_save, gflops );

        }
    
        free(a);
        free(b);
        free(c);
        free(c_save);
	}

    MPI_Finalize();

	return 0;
}

