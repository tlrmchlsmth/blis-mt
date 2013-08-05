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

void bli_trmm_lu_blk_var4( obj_t*  alpha,
                           obj_t*  a,
                           obj_t*  b,
                           obj_t*  beta,
                           obj_t*  c,
                           trmm_t* cntl )
{
	obj_t a1, a1_pack;
	obj_t b_pack;
	obj_t c1, c1_pack;

	dim_t i;
	dim_t bm_alg;
	dim_t mT_trans;

	// Initialize all pack objects that are passed into packm_init().
	bli_obj_init_pack( &a1_pack );
	bli_obj_init_pack( &b_pack );
	bli_obj_init_pack( &c1_pack );

	// Query dimension in partitioning direction. Use the diagonal offset
	// to stop short of the zero region.
	mT_trans = bli_abs( bli_obj_diag_offset_after_trans( *a ) ) +
	           bli_obj_width_after_trans( *a );

	// Scale C by beta (if instructed).
	bli_scalm_int( beta,
	               c,
	               cntl_sub_scalm( cntl ) );

	// Initialize object for packing B.
	bli_packm_init( b, &b_pack,
	                cntl_sub_packm_b( cntl ) );

	// Fuse the first iteration with incremental packing and computation.
	{
		obj_t b_inc, b_pack_inc;
		obj_t c1_pack_inc;

		dim_t j;
		dim_t bn_inc;
		dim_t n_trans;

		// Query dimension in partitioning direction.
		n_trans = bli_obj_width( b_pack );

		// Determine the current algorithmic blocksize.
		bm_alg = bli_determine_blocksize_f( 0, mT_trans, a,
		                                    cntl_blocksize( cntl ) );

		// Acquire partitions for A1 and C1.
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       0, bm_alg, a, &a1 );
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       0, bm_alg, c, &c1 );

		// Initialize objects for packing A1 and C1.
		bli_packm_init( &a1, &a1_pack, cntl_sub_packm_a( cntl ) );
		bli_packm_init( &c1, &c1_pack, cntl_sub_packm_c( cntl ) );

		// Pack A1 and scale by alpha (if instructed).
		bli_packm_int( alpha, &a1, &a1_pack, cntl_sub_packm_a( cntl ) );

		// Pack C1 and scale by beta (if instructed).
		bli_packm_int( beta,  &c1, &c1_pack, cntl_sub_packm_c( cntl ) );

		// Partition along the n dimension.
		for ( j = 0; j < n_trans; j += bn_inc )
		{
			// Determine the current incremental packing blocksize.
			bn_inc = bli_determine_blocksize_f( j, n_trans, b,
			                                    cntl_blocksize_aux( cntl ) );

			// Acquire partitions.
			bli_acquire_mpart_l2r( BLIS_SUBPART1,
			                       j, bn_inc, b, &b_inc );
			bli_acquire_mpart_l2r( BLIS_SUBPART1,
			                       j, bn_inc, &b_pack, &b_pack_inc );
			bli_acquire_mpart_l2r( BLIS_SUBPART1,
			                       j, bn_inc, &c1_pack, &c1_pack_inc );

			// Pack B1 and scale by alpha (if instructed).
			bli_packm_int( alpha, &b_inc, &b_pack_inc, cntl_sub_packm_b( cntl ) );

			// Perform trmm subproblem.
			bli_trmm_int( BLIS_LEFT,
			              alpha,
			              &a1_pack,
			              &b_pack_inc,
			              beta,
			              &c1_pack_inc,
			              cntl_sub_trmm( cntl ) );
		}

		// Unpack C1 (if C1 was packed).
		bli_unpackm_int( &c1_pack, &c1, cntl_sub_unpackm_c( cntl ) );
	}


	// Partition along the remaining portion of the m dimension.
	for ( i = bm_alg; i < mT_trans; i += bm_alg )
	{
		// Determine the current algorithmic blocksize.
		bm_alg = bli_determine_blocksize_f( i, mT_trans, a,
		                                    cntl_blocksize( cntl ) );

		// Acquire partitions for A1 and C1.
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       i, bm_alg, a, &a1 );
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       i, bm_alg, c, &c1 );

		// Initialize objects for packing A1 and C1.
		bli_packm_init( &a1, &a1_pack,
		                cntl_sub_packm_a( cntl ) );
		bli_packm_init( &c1, &c1_pack,
		                cntl_sub_packm_c( cntl ) );

		// Pack A1 and scale by alpha (if instructed).
		bli_packm_int( alpha,
		               &a1, &a1_pack,
		               cntl_sub_packm_a( cntl ) );

		// Pack C1 and scale by beta (if instructed).
		bli_packm_int( beta,
		               &c1, &c1_pack,
		               cntl_sub_packm_c( cntl ) );

		// Perform trmm subproblem.
		if ( bli_obj_intersects_diag( a1_pack ) )
			bli_trmm_int( BLIS_LEFT,
			              alpha,
			              &a1_pack,
			              &b_pack,
			              beta,
			              &c1_pack,
			              cntl_sub_trmm( cntl ) );
		else
			bli_gemm_int( alpha,
			              &a1_pack,
			              &b_pack,
			              &BLIS_ONE,
			              &c1_pack,
			              cntl_sub_gemm( cntl ) );

		// Unpack C1 (if C1 was packed).
		bli_unpackm_int( &c1_pack, &c1,
		                 cntl_sub_unpackm_c( cntl ) );
	}

	// If any packing buffers were acquired within packm, release them back
	// to the memory manager.
	bli_obj_release_pack( &a1_pack );
	bli_obj_release_pack( &b_pack );
	bli_obj_release_pack( &c1_pack );
}

