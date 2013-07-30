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

#include "blis.h"

void bli_gemm_blk_var2( obj_t*  alpha,
                        obj_t*  a,
                        obj_t*  b,
                        obj_t*  beta,
                        obj_t*  c,
                        gemm_t* cntl )
{
	obj_t a_pack_s;
	obj_t b1_pack_s;
	obj_t c1_pack_s;

    obj_t  b1, c1;
    obj_t* a_pack   = NULL;
    obj_t* b1_pack  = NULL;
    obj_t* c1_pack  = NULL;

	dim_t i;
	dim_t b_alg;
	dim_t n_trans;
    
    dim_t num_groups = bli_gemm_num_thread_groups( cntl->thread_info );
    dim_t group_id   = bli_gemm_group_id( cntl->thread_info );

    if( bli_gemm_am_a_master( cntl->thread_info ) ) {
        // Initialize object for packing A.
        bli_obj_init_pack( &a_pack_s );
        bli_packm_init( a, &a_pack_s,
                        cntl_sub_packm_a( cntl ) );

    }
    a_pack = bli_gemm_broadcast_a( cntl->thread_info, &a_pack_s );

	// Pack A and scale by alpha (if instructed).
	bli_packm_int( alpha,
	               a, a_pack,
	               cntl_sub_packm_a( cntl ) );

    bli_gemm_a_barrier( cntl->thread_info );

	if( bli_gemm_am_b_master( cntl->thread_info )) {
        bli_obj_init_pack( &b1_pack_s );
    }
    b1_pack  = bli_gemm_broadcast_b( cntl->thread_info, &b1_pack_s );

    if( bli_gemm_am_c_master( cntl->thread_info )) {
        bli_obj_init_pack( &c1_pack_s );

        // Scale C by beta (if instructed).
        bli_scalm_int( beta,
                       c,
                       cntl_sub_scalm( cntl ) );
    }
    c1_pack  = bli_gemm_broadcast_c( cntl->thread_info, &c1_pack_s );


	// Query dimension in partitioning direction.
	n_trans = bli_obj_width_after_trans( *b );
    dim_t n_pt = n_trans / num_groups;
    n_pt = (n_pt * num_groups < n_trans) ? n_pt + 1 : n_pt;
    n_pt = (n_pt % 8 == 0) ? n_pt : n_pt + 8 - (n_pt % 8);
    dim_t start = group_id * n_pt;
    dim_t end = bli_min( start + n_pt, n_trans );

	// Partition along the n dimension.
	for ( i = start; i < end; i += b_alg )
	{
		// Determine the current algorithmic blocksize.
		// NOTE: Use of b (for execution datatype) is intentional!
		// This causes the right blocksize to be used if c and a are
		// complex and b is real.
		b_alg = bli_determine_blocksize_f( i, end, b,
		                                   cntl_blocksize( cntl ) );

        // Acquire partitions for C1 
        bli_acquire_mpart_l2r( BLIS_SUBPART1,
                               i, b_alg, c, &c1 );
        // Acquire partitions for B1 
        bli_acquire_mpart_l2r( BLIS_SUBPART1,
                               i, b_alg, b, &b1 );

        if( bli_gemm_am_b_master( cntl->thread_info )) {
            // Initialize objects for packing B1 
            bli_packm_init( &b1, &b1_pack_s,
                            cntl_sub_packm_b( cntl ) );
        }

        if( bli_gemm_am_c_master( cntl->thread_info )) {
            // Initialize objects for packing C1 
            bli_packm_init( &c1, &c1_pack_s,
                            cntl_sub_packm_c( cntl ) );
        }

        bli_gemm_b_barrier( cntl->thread_info );
        bli_gemm_c_barrier( cntl->thread_info );
        
		// Pack B1 and scale by alpha (if instructed).
		bli_packm_int( alpha,
		               &b1, b1_pack,
		               cntl_sub_packm_b( cntl ) );

		// Pack C1 and scale by beta (if instructed).
		bli_packm_int( beta,
		               &c1, c1_pack,
		               cntl_sub_packm_c( cntl ) );

        // Packing must be done before computation
        bli_gemm_b_barrier( cntl->thread_info );
        bli_gemm_c_barrier( cntl->thread_info );

		// Perform gemm subproblem.
		bli_gemm_int( alpha,
		              a_pack,
		              b1_pack,
		              beta,
		              c1_pack,
		              cntl_sub_gemm( cntl ) );

		// Unpack C1 (if C1 was packed).
		bli_unpackm_int( c1_pack, &c1,
		                 cntl_sub_unpackm_c( cntl ) );
	}

	// If any packing buffers were acquired within packm, release them back
	// to the memory manager.
    bli_gemm_a_barrier( cntl->thread_info );
    if( bli_gemm_am_a_master( cntl->thread_info ))
	    bli_obj_release_pack( &a_pack_s );
    bli_gemm_b_barrier( cntl->thread_info );
    if( bli_gemm_am_b_master( cntl->thread_info )) {
        bli_obj_release_pack( &b1_pack_s );
    }
    bli_gemm_c_barrier( cntl->thread_info );
    if( bli_gemm_am_c_master( cntl->thread_info )) {
        bli_obj_release_pack( &c1_pack_s );
    }
}

