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

void bli_trsm_blk_var1f( obj_t*  alpha,
                         obj_t*  a,
                         obj_t*  b,
                         obj_t*  beta,
                         obj_t*  c,
                         trsm_t* cntl )
{
	obj_t a1, a1_pack;
	obj_t b_pack;
	obj_t c1;

	dim_t i;
	dim_t b_alg;
	dim_t m_trans;
	dim_t offA;

	// Initialize all pack objects that are passed into packm_init().
	bli_obj_init_pack( &a1_pack );
	bli_obj_init_pack( &b_pack );

	// Set the default length of and offset to the non-zero part of A.
	m_trans  = bli_obj_length_after_trans( *a );
	offA     = 0;

	// If A is lower triangular, we have to adjust where the non-zero part of
	// A begins.
	if ( bli_obj_is_lower( *a ) )
		offA = bli_abs( bli_obj_diag_offset_after_trans( *a ) );

	// Initialize object for packing B.
	bli_packm_init( b, &b_pack,
	                cntl_sub_packm_b( cntl ) );

	// Pack B1 and scale by alpha (if instructed).
	bli_packm_int( alpha,
	               b, &b_pack,
	               cntl_sub_packm_b( cntl ) );

	// Partition along the remaining portion of the m dimension.
	for ( i = offA; i < m_trans; i += b_alg )
	{
		// Determine the current algorithmic blocksize.
		b_alg = bli_determine_blocksize_f( i, m_trans, a,
		                                   cntl_blocksize( cntl ) );

		// Acquire partitions for A1 and C1.
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       i, b_alg, a, &a1 );
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       i, b_alg, c, &c1 );

		// Initialize object for packing A1.
		bli_packm_init( &a1, &a1_pack,
		                cntl_sub_packm_a( cntl ) );

		// Pack A1 and scale by alpha (if instructed).
		bli_packm_int( alpha,
		               &a1, &a1_pack,
		               cntl_sub_packm_a( cntl ) );

		// Perform trsm subproblem.
		bli_trsm_int( alpha,
		              &a1_pack,
		              &b_pack,
		              beta,
		              &c1,
		              cntl_sub_trsm( cntl ) );
	}

	// If any packing buffers were acquired within packm, release them back
	// to the memory manager.
	bli_obj_release_pack( &a1_pack );
	bli_obj_release_pack( &b_pack );
}

