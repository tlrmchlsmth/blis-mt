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

void bli_gemm_blk_var1( obj_t*  alpha,
                        obj_t*  a,
                        obj_t*  b,
                        obj_t*  beta,
                        obj_t*  c,
                        gemm_t* cntl )
{
	obj_t a1_pack_s;
	obj_t b_pack_s;
	obj_t c1_pack_s;

    obj_t a1, c1;
    obj_t* a1_pack  = NULL;
    obj_t* b_pack   = NULL;
    obj_t* c1_pack  = NULL;

	dim_t i;
	dim_t b_alg;
	dim_t m_trans;
    
    dim_t num_groups = bli_gemm_num_thread_groups( cntl->thread_info );
    dim_t group_id = bli_gemm_group_id( cntl->thread_info );

    if( bli_gemm_am_b_master( cntl->thread_info ) ) {
        bli_obj_init_pack( &b_pack_s );

        bli_packm_init( b, &b_pack_s,
                        cntl_sub_packm_b( cntl ));
    }
    b_pack = bli_gemm_broadcast_b( cntl->thread_info, &b_pack_s );
    bli_gemm_b_barrier( cntl->thread_info );

    if( bli_gemm_am_c_master( cntl->thread_info ) ) {
        bli_obj_init_pack( &c1_pack_s );

        bli_scalm_int( beta,
                       c,
                       cntl_sub_scalm( cntl ));
    }
    c1_pack = bli_gemm_broadcast_c( cntl->thread_info, &c1_pack_s );

    if( bli_gemm_am_a_master( cntl->thread_info )) {
        bli_obj_init_pack( &a1_pack_s );
    }
    a1_pack = bli_gemm_broadcast_a( cntl->thread_info, &a1_pack_s );

	// Pack B and scale by alpha (if instructed).
	bli_packm_int( alpha,
	               b, b_pack,
	               cntl_sub_packm_b( cntl ) );


	// Query dimension in partitioning direction.
	m_trans = bli_obj_length_after_trans( *a );
    dim_t m_pt = m_trans / num_groups;
    m_pt = (m_pt * num_groups < m_trans) ? m_pt + 1 : m_pt;
    m_pt = (m_pt % 8 == 0) ? m_pt : m_pt + 8 - (m_pt % 8);
    dim_t start = group_id * m_pt;
    dim_t end = bli_min( start + m_pt, m_trans );
	//printf("%d\t%d\t%d\t%d\n", m_trans, m_pt, start, end );

    // Partition along the m dimension.
	for ( i = start; i < end; i += b_alg )
	{
		// Determine the current algorithmic blocksize.
		// NOTE: Use of a (for execution datatype) is intentional!
		// This causes the right blocksize to be used if c and a are
		// complex and b is real.
		b_alg = bli_determine_blocksize_f( i, end, a,
		                                   cntl_blocksize( cntl ));

        // Acquire partitions for A1 
        bli_acquire_mpart_t2b( BLIS_SUBPART1,
                               i, b_alg, a, &a1 );

        // Acquire partitions for C1 
        bli_acquire_mpart_t2b( BLIS_SUBPART1,
                               i, b_alg, c, &c1 );

        if( bli_gemm_am_a_master( cntl->thread_info )) {
            // Initialize objects for packing A1 
            bli_packm_init( &a1, &a1_pack_s,
                            cntl_sub_packm_a( cntl ));
        }

        if( bli_gemm_am_c_master( cntl->thread_info )) {
            // Initialize objects for packing A1 
            bli_packm_init( &c1, &c1_pack_s,
                            cntl_sub_packm_c( cntl ));
        }

        bli_gemm_a_barrier( cntl->thread_info );
        bli_gemm_c_barrier( cntl->thread_info );

        //printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n", omp_get_thread_num(), 1, a1, a1_pack, c1, c1_pack, b_pack);

        // Pack A1 and scale by alpha (if instructed).
		bli_packm_int( alpha,
		               &a1, a1_pack,
		               cntl_sub_packm_a( cntl ) );

		// Pack C1 and scale by beta (if instructed).
		bli_packm_int( beta,
		               &c1, c1_pack,
		               cntl_sub_packm_c( cntl ) );

        // Packing must be done before computation is done.
        bli_gemm_a_barrier( cntl->thread_info );
        bli_gemm_c_barrier( cntl->thread_info );

		// Perform gemm subproblem.
		bli_gemm_int( alpha,
		              a1_pack,
		              b_pack,
		              beta,
		              c1_pack,
		              cntl_sub_gemm( cntl ) );

		// Unpack C1 (if C1 was packed).
		bli_unpackm_int( c1_pack, &c1,
		                 cntl_sub_unpackm_c( cntl ));
	}

    // If any packing buffers were acquired within packm, release them back
    // to the memory manager.
    bli_gemm_b_barrier( cntl->thread_info );
    if( bli_gemm_am_b_master( cntl->thread_info ))
        bli_obj_release_pack( &b_pack_s );
    bli_gemm_a_barrier( cntl->thread_info );
    if( bli_gemm_am_a_master( cntl->thread_info )) {
        bli_obj_release_pack( &a1_pack_s );
    }
    bli_gemm_c_barrier( cntl->thread_info );
    if( bli_gemm_am_c_master( cntl->thread_info )) {
        bli_obj_release_pack( &c1_pack_s );
    }
}
