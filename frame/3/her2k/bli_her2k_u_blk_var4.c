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

void bli_her2k_u_blk_var4( obj_t*   alpha,
                           obj_t*   a,
                           obj_t*   bh,
                           obj_t*   alpha_conj,
                           obj_t*   b,
                           obj_t*   ah,
                           obj_t*   beta,
                           obj_t*   c,
                           her2k_t* cntl )
{
	obj_t a1, a1_pack;
	obj_t bh_pack, bhR_pack;
	obj_t b1, b1_pack;
	obj_t ah_pack, ahR_pack;
	obj_t c1, c1_pack;
	obj_t c1R, c1R_pack;

	dim_t i;
	dim_t bm_alg;
	dim_t m_trans;
	dim_t offR, nR;

	// Initialize all pack objects that are passed into packm_init().
	bli_obj_init_pack( &a1_pack );
	bli_obj_init_pack( &bh_pack );
	bli_obj_init_pack( &b1_pack );
	bli_obj_init_pack( &ah_pack );
	bli_obj_init_pack( &c1_pack );
	bli_obj_init_pack( &c1R_pack );

	// Query dimension in partitioning direction.
	m_trans = bli_obj_length_after_trans( *c );

	// Scale C by beta (if instructed).
	bli_scalm_int( beta,
	               c,
	               cntl_sub_scalm( cntl ) );

	// Initialize object for packing B1'.
	bli_packm_init( bh, &bh_pack,
	                cntl_sub_packm_b( cntl ) );

	// Initialize object for packing A1'.
	bli_packm_init( ah, &ah_pack,
	                cntl_sub_packm_b( cntl ) );

	// Fuse the first iteration with incremental packing and computation.
	{
		obj_t bh_inc, bh_pack_inc;
		obj_t ah_inc, ah_pack_inc;
		obj_t c1_pack_inc;

		dim_t j;
		dim_t bn_inc;
		dim_t n_trans;

		// Query dimension in partitioning direction.
		n_trans = bli_obj_width( bh_pack );

		// Determine the current algorithmic blocksize.
		bm_alg = bli_determine_blocksize_f( 0, m_trans, a,
		                                    cntl_blocksize( cntl ) );

		// Acquire partitions for A1, B1, and C1.
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       0, bm_alg, a, &a1 );
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       0, bm_alg, b, &b1 );
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       0, bm_alg, c, &c1 );

		// Initialize objects for packing A1, B1, and C1.
		bli_packm_init( &a1, &a1_pack, cntl_sub_packm_a( cntl ) );
		bli_packm_init( &b1, &b1_pack, cntl_sub_packm_a( cntl ) );
		bli_packm_init( &c1, &c1_pack, cntl_sub_packm_c( cntl ) );

		// Pack C1 and scale by beta (if instructed).
		bli_packm_int( beta,  &c1, &c1_pack, cntl_sub_packm_c( cntl ) );

		// Pack A1 and scale by alpha (if instructed).
		bli_packm_int( alpha, &a1, &a1_pack, cntl_sub_packm_a( cntl ) );

		// Partition along the n dimension.
		for ( j = 0; j < n_trans; j += bn_inc )
		{
			// Determine the current incremental packing blocksize.
			bn_inc = bli_determine_blocksize_f( j, n_trans, a,
			                                    cntl_blocksize_aux( cntl ) );

			// Acquire incremental partitions.
			bli_acquire_mpart_l2r( BLIS_SUBPART1,
			                       j, bn_inc, bh, &bh_inc );
			bli_acquire_mpart_l2r( BLIS_SUBPART1,
			                       j, bn_inc, &bh_pack, &bh_pack_inc );
			bli_acquire_mpart_l2r( BLIS_SUBPART1,
			                       j, bn_inc, &c1_pack, &c1_pack_inc );

			// Pack Bh_inc and scale by alpha (if instructed).
			bli_packm_int( alpha, &bh_inc, &bh_pack_inc, cntl_sub_packm_b( cntl ) );

			// Perform herk subproblem.
			bli_herk_int( &BLIS_ONE,
			              &a1_pack,
			              &bh_pack_inc,
			              beta,
			              &c1_pack_inc,
			              cntl_sub_herk( cntl ) );
		}
		
		// Pack B1 and scale by alpha_conj (if instructed).
		bli_packm_int( alpha_conj, &b1, &b1_pack, cntl_sub_packm_a( cntl ) );

		// Partition along the n dimension.
		for ( j = 0; j < n_trans; j += bn_inc )
		{
			// Determine the current incremental packing blocksize.
			bn_inc = bli_determine_blocksize_f( j, n_trans, b,
			                                    cntl_blocksize_aux( cntl ) );

			// Acquire incremental partitions.
			bli_acquire_mpart_l2r( BLIS_SUBPART1,
			                       j, bn_inc, ah, &ah_inc );
			bli_acquire_mpart_l2r( BLIS_SUBPART1,
			                       j, bn_inc, &ah_pack, &ah_pack_inc );
			bli_acquire_mpart_l2r( BLIS_SUBPART1,
			                       j, bn_inc, &c1_pack, &c1_pack_inc );

			// Pack Ah_inc and scale by alpha_conj (if instructed).
			bli_packm_int( alpha_conj, &ah_inc, &ah_pack_inc, cntl_sub_packm_b( cntl ) );

			// Perform herk subproblem.
			bli_herk_int( &BLIS_ONE,
			              &b1_pack,
			              &ah_pack_inc,
			              beta,
			              &c1_pack_inc,
			              cntl_sub_herk( cntl ) );
		}
		
		// Unpack C1 (if C1 was packed).
		bli_unpackm_int( &c1_pack, &c1, cntl_sub_unpackm_c( cntl ) );
	}

	// Partition along the m dimension.
	for ( i = bm_alg; i < m_trans; i += bm_alg )
	{
		// Determine the current algorithmic blocksize.
		bm_alg = bli_determine_blocksize_f( i, m_trans, a,
		                                    cntl_blocksize( cntl ) );

		// Acquire partitions for A1, B1, and C1.
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       i, bm_alg, a, &a1 );
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       i, bm_alg, b, &b1 );
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       i, bm_alg, c, &c1 );

		// Partition off the stored region of C1 and the corresponding regions
		// of Bh_pack and Ah_pack. We compute the width of the subpartition
		// taking the location of the diagonal into account.
		offR = bli_max( 0, bli_obj_diag_offset_after_trans( c1 ) );
		nR   = bli_obj_width_after_trans( c1 ) - offR;
		bli_acquire_mpart_l2r( BLIS_SUBPART1,
		                       offR, nR, &c1, &c1R );
		bli_acquire_mpart_l2r( BLIS_SUBPART1,
		                       offR, nR, &bh_pack, &bhR_pack );
		bli_acquire_mpart_l2r( BLIS_SUBPART1,
		                       offR, nR, &ah_pack, &ahR_pack );

		// Initialize objects for packing A1, B1, and C1.
		bli_packm_init( &a1, &a1_pack,
		                cntl_sub_packm_a( cntl ) );
		bli_packm_init( &b1, &b1_pack,
		                cntl_sub_packm_a( cntl ) );
		bli_packm_init( &c1R, &c1R_pack,
		                cntl_sub_packm_c( cntl ) );

		// Pack A1 and scale by alpha (if instructed).
		bli_packm_int( alpha,
		               &a1, &a1_pack,
		               cntl_sub_packm_a( cntl ) );

		// Pack B1 and scale by alpha_conj (if instructed).
		bli_packm_int( alpha_conj,
		               &b1, &b1_pack,
		               cntl_sub_packm_a( cntl ) );

		// Pack C1 and scale by beta (if instructed).
		bli_packm_int( beta,
		               &c1R, &c1R_pack,
		               cntl_sub_packm_c( cntl ) );

		// Perform herk subproblem.
		bli_her2k_int( alpha,
		               &a1_pack,
		               &bhR_pack,
		               alpha_conj,
		               &b1_pack,
		               &ahR_pack,
		               beta,
		               &c1R_pack,
		               cntl_sub_her2k( cntl ) );

		// Unpack C1 (if C1 was packed).
		bli_unpackm_int( &c1R_pack, &c1R,
		                 cntl_sub_unpackm_c( cntl ) );
	}

	// If any packing buffers were acquired within packm, release them back
	// to the memory manager.
	bli_obj_release_pack( &a1_pack );
	bli_obj_release_pack( &bh_pack );
	bli_obj_release_pack( &b1_pack );
	bli_obj_release_pack( &ah_pack );
	bli_obj_release_pack( &c1_pack );
	bli_obj_release_pack( &c1R_pack );
}

