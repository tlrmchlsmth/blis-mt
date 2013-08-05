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

void reduce( obj_t* c, obj_t* others, int n_to_reduce, int n_threads, int t_id )
{
    if(n_to_reduce <= 1)
        return;

    int rs = bli_obj_row_stride( *c );
    int cs = bli_obj_col_stride( *c );
    int m = bli_obj_length( others[0] );
    int n = bli_obj_width( others[0] );

    double * cp = (double *)bli_obj_buffer_at_off( *c );
    
    for( int j = t_id; j < n; j += n_threads )
    {   
        for( int i = 0; i < m; i++ )
        {   
            double sum = 0.0;
            for( int k = 0; k < n_to_reduce - 1; k++ )
            {   
                double * op = (double *) bli_obj_buffer_at_off( others[k] );
                sum += op[j*cs+i*rs];
            }   
            cp[j*cs+i*rs] += sum;
        }   
    }   
}


void bli_gemm_blk_var3( obj_t*  alpha,
                        obj_t*  a,
                        obj_t*  b,
                        obj_t*  beta,
                        obj_t*  c,
                        gemm_t* cntl )
{
	obj_t  a1_pack_s;
	obj_t  b1_pack_s;
	obj_t  c_pack_s;
     
    obj_t a1, b1;
	obj_t* a1_pack = NULL;
	obj_t* b1_pack = NULL;
	obj_t* c_pack  = NULL;

    obj_t* c_tmps  = NULL;
    obj_t* my_c    = NULL;

    thread_comm_t* group_communicators = NULL;
    thread_comm_t* group_communicator = NULL;

	obj_t* beta_use;

	dim_t  i;
	dim_t  b_alg;
	dim_t  k_trans;

    dim_t  group_id = bli_gemm_group_id( cntl->thread_info );
    dim_t  num_groups = bli_gemm_num_thread_groups( cntl->thread_info );
    dim_t  threads_per_group = bli_gemm_c_num_threads( cntl->thread_info ) / num_groups;
    dim_t  id_within_group = bli_gemm_c_id( cntl->thread_info ) % threads_per_group;

    if( bli_gemm_am_c_master( cntl->thread_info )) {
        // Setup communicators for within groups.
        group_communicators = (thread_comm_t*) bli_malloc( sizeof(thread_comm_t) * num_groups );

        // Setup temporary c's for within groups
        if( num_groups > 1 ) {
            c_tmps = (obj_t*) bli_malloc( sizeof(obj_t) * (num_groups - 1) );
        }

        // Scale C by beta (if instructed).
        bli_scalm_int( beta, c, cntl_sub_scalm( cntl ) );
    }
    group_communicators = bli_gemm_broadcast_c( cntl->thread_info, group_communicators );
    group_communicator  = &group_communicators[ group_id ];
    c_tmps = bli_gemm_broadcast_c( cntl->thread_info, c_tmps );
    bli_gemm_c_barrier( cntl->thread_info );

    if( !group_id )    my_c = c;
    else               my_c = &c_tmps[ group_id - 1 ];

    if( bli_gemm_am_a_master( cntl->thread_info ) ) {
        // Initialize a1_pack
        bli_obj_init_pack( &a1_pack_s );
    }
    a1_pack = bli_gemm_broadcast_a( cntl->thread_info, &a1_pack_s );

    if( bli_gemm_am_b_master( cntl->thread_info ) ) {
        bli_obj_init_pack( &b1_pack_s );
    }
    b1_pack = bli_gemm_broadcast_b( cntl->thread_info, &b1_pack_s );

    if( id_within_group == 0 ) {
        // Initialize c1_pack
        bli_obj_init_pack( &c_pack_s );

        // Setup group communicator.
        bli_setup_communicator( group_communicator, threads_per_group );

        // Initialize this thread group's temp C, (if my_c != c)
        if( group_id ) {
            bli_obj_create( bli_obj_datatype( *c ), c->m, c->n, c->rs, c->cs, my_c );
            bli_setm( &BLIS_ZERO, my_c );
        }

        // Initialize object for packing C.
        bli_packm_init( my_c, &c_pack_s,
                        cntl_sub_packm_c( cntl ) );
    }
    bli_gemm_c_barrier( cntl->thread_info );
    c_pack = bli_broadcast_structure( group_communicator, id_within_group, &c_pack_s );
	
    // Pack C and scale by beta (if instructed).
	bli_packm_int( beta,
	               my_c, c_pack,
	               cntl_sub_packm_c( cntl ) );

	// Query dimension in partitioning direction.
	k_trans = bli_obj_width_after_trans( *a );
    dim_t k_pt = k_trans / num_groups;
    k_pt = (k_pt * num_groups < k_trans) ? k_pt + 1 : k_pt;
    dim_t start = group_id * k_pt;
    dim_t end = bli_min( start + k_pt, k_trans );

//    printf("%d\t%d\t%d\t%d\n", k_pt, start, end, group_id);
	
    // Partition along the k dimension.
	for ( i = start; i < end; i += b_alg )
	{
		// Determine the current algorithmic blocksize.
		// NOTE: Use of b (for execution datatype) is intentional!
		// This causes the right blocksize to be used if c and a are
		// complex and b is real.
		b_alg = bli_determine_blocksize_f( i, end, b,
		                                   cntl_blocksize( cntl ) );

        // Acquire partitions for A1 
        bli_acquire_mpart_l2r( BLIS_SUBPART1,
                               i, b_alg, a, &a1 );
        // Acquire partitions for B1.
        bli_acquire_mpart_t2b( BLIS_SUBPART1,
                               i, b_alg, b, &b1 );
        
        if( bli_gemm_am_a_master( cntl->thread_info )) {
            // Initialize objects for packing A1 
            bli_packm_init( &a1, &a1_pack_s,
                            cntl_sub_packm_a( cntl ) );
        }

        if( bli_gemm_am_b_master( cntl->thread_info )) {
            // Initialize objects for packing B1.
            bli_packm_init( &b1, &b1_pack_s,
                            cntl_sub_packm_b( cntl ) );
        }

        //printf("%d\t%d\t%d\t%d\n", a1, b1, a1_pack, b1_pack);   
        bli_gemm_a_barrier( cntl->thread_info );
        bli_gemm_b_barrier( cntl->thread_info );

		// Pack A1 and scale by alpha (if instructed).
		bli_packm_int( alpha,
		               &a1, a1_pack,
		               cntl_sub_packm_a( cntl ) );

		// Pack B1 and scale by alpha (if instructed).
		bli_packm_int( alpha,
		               &b1, b1_pack,
		               cntl_sub_packm_b( cntl ) );

		// Since this variant executes multiple rank-k updates, we must use
		// beta only for the first iteration and BLIS_ONE for all others.
		if ( i == 0 ) beta_use = beta;
		else          beta_use = &BLIS_ONE;

        // Packing must be done before computation is done.
        bli_gemm_a_barrier( cntl->thread_info );
        bli_gemm_b_barrier( cntl->thread_info );

		// Perform gemm subproblem.
		bli_gemm_int( alpha,
		              a1_pack,
		              b1_pack,
		              beta_use,
		              c_pack,
		              cntl_sub_gemm( cntl ) );  

	}
    
    bli_gemm_c_barrier( cntl->thread_info );
    reduce( c, c_tmps, num_groups,
        bli_gemm_c_num_threads( cntl->thread_info ), 
        bli_gemm_c_id( cntl->thread_info ) );  
    bli_gemm_c_barrier( cntl->thread_info );

    if( id_within_group == 0 ) {
        // Unpack C (if C was packed).
        bli_unpackm_int( c_pack, my_c,
                         cntl_sub_unpackm_c( cntl ) );
    }

	// If any packing buffers were acquired within packm, release them back
	// to the memory manager.
    bli_gemm_a_barrier( cntl->thread_info );
    if( bli_gemm_am_a_master( cntl->thread_info )) {
        bli_obj_release_pack( &a1_pack_s );
    }
    bli_gemm_b_barrier( cntl->thread_info );
    if( bli_gemm_am_b_master( cntl->thread_info )) {
        bli_obj_release_pack( &b1_pack_s );
    }
    bli_gemm_c_barrier( cntl->thread_info );
    if( id_within_group == 0 ) {
       // bli_cleanup_communicator( group_communicator );
	    bli_obj_release_pack( c_pack );
        if( c != my_c )
            bli_obj_free( my_c );
    }
    if( bli_gemm_am_c_master( cntl->thread_info ) && num_groups > 1) {
            bli_free( c_tmps );
            bli_free( group_communicators );
    }
}
