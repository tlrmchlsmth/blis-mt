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

extern scalm_t*   scalm_cntl;

gemm_t*           gemm_cntl;
gemm_t*           gemm_cntl_packa;

void bli_gemm_hier_cntl_create();
void bli_gemm_grid_cntl_create();

bool_t do_gridlike = 0;

dim_t l0_nt = BLIS_GEMM_UKERNEL_THREADS; // Number of threads used in the microkernel. Currently not used in the gridlike threading.
dim_t l1_nt = 1; 
dim_t l2_nt = 1;
dim_t l3_nt = 1;
dim_t l4_nt = 1;
dim_t l5_nt = 1;
dim_t max_pack_with = 64;


gemm_t**          gemm_cntl_mts;
dim_t             gemm_num_threads_default;

gemm_t*           gemm_cntl_bp_ke;
gemm_t*           gemm_cntl_op_bp;
gemm_t*           gemm_cntl_mm_op;
gemm_t*           gemm_cntl_vl_mm;

gemm_t*           gemm_cntl_bp_ke5;
gemm_t*           gemm_cntl_pm_bp;
gemm_t*           gemm_cntl_mm_pm;
gemm_t*           gemm_cntl_vl_mm5;

packm_t*          gemm_packa_cntl;
packm_t*          gemm_packb_cntl;
packm_t*          gemm_packc_cntl;
unpackm_t*        gemm_unpackc_cntl;

blksz_t*          gemm_mc;
blksz_t*          gemm_nc;
blksz_t*          gemm_kc;
blksz_t*          gemm_mr;
blksz_t*          gemm_nr;
blksz_t*          gemm_kr;
blksz_t*          gemm_ni;


void bli_gemm_cntl_init()
{
	// Create blocksize objects for each dimension.
	gemm_mc = bli_blksz_obj_create( BLIS_DEFAULT_MC_S, BLIS_EXTEND_MC_S,
	                                BLIS_DEFAULT_MC_D, BLIS_EXTEND_MC_D,
	                                BLIS_DEFAULT_MC_C, BLIS_EXTEND_MC_C,
	                                BLIS_DEFAULT_MC_Z, BLIS_EXTEND_MC_Z );

	gemm_nc = bli_blksz_obj_create( BLIS_DEFAULT_NC_S, BLIS_EXTEND_NC_S,
	                                BLIS_DEFAULT_NC_D, BLIS_EXTEND_NC_D,
	                                BLIS_DEFAULT_NC_C, BLIS_EXTEND_NC_C,
	                                BLIS_DEFAULT_NC_Z, BLIS_EXTEND_NC_Z );

	gemm_kc = bli_blksz_obj_create( BLIS_DEFAULT_KC_S, BLIS_EXTEND_KC_S,
	                                BLIS_DEFAULT_KC_D, BLIS_EXTEND_KC_D,
	                                BLIS_DEFAULT_KC_C, BLIS_EXTEND_KC_C,
	                                BLIS_DEFAULT_KC_Z, BLIS_EXTEND_KC_Z );

	gemm_mr = bli_blksz_obj_create( BLIS_DEFAULT_MR_S, BLIS_EXTEND_MR_S,
	                                BLIS_DEFAULT_MR_D, BLIS_EXTEND_MR_D,
	                                BLIS_DEFAULT_MR_C, BLIS_EXTEND_MR_C,
	                                BLIS_DEFAULT_MR_Z, BLIS_EXTEND_MR_Z );

	gemm_nr = bli_blksz_obj_create( BLIS_DEFAULT_NR_S, BLIS_EXTEND_NR_S,
	                                BLIS_DEFAULT_NR_D, BLIS_EXTEND_NR_D,
	                                BLIS_DEFAULT_NR_C, BLIS_EXTEND_NR_C,
	                                BLIS_DEFAULT_NR_Z, BLIS_EXTEND_NR_Z );

	gemm_kr = bli_blksz_obj_create( BLIS_DEFAULT_KR_S, BLIS_EXTEND_KR_S,
	                                BLIS_DEFAULT_KR_D, BLIS_EXTEND_KR_D,
	                                BLIS_DEFAULT_KR_C, BLIS_EXTEND_KR_C,
	                                BLIS_DEFAULT_KR_Z, BLIS_EXTEND_KR_Z );

	gemm_ni = bli_blksz_obj_create( BLIS_DEFAULT_NI_S, 0,
	                                BLIS_DEFAULT_NI_D, 0,
	                                BLIS_DEFAULT_NI_C, 0,
	                                BLIS_DEFAULT_NI_Z, 0 );


	// Create control tree objects for packm operations.
	gemm_packa_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           gemm_mr,
	                           gemm_kr,
	                           FALSE, // do NOT scale by alpha
	                           FALSE, // already dense; densify not necessary
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_ROW_PANELS,
	                           BLIS_BUFFER_FOR_A_BLOCK );

	gemm_packb_cntl
	=
	bli_packm_cntl_obj_create( BLIS_BLOCKED,
	                           BLIS_VARIANT2,
	                           gemm_kr,
	                           gemm_nr,
	                           FALSE, // do NOT scale by alpha
	                           FALSE, // already dense; densify not necessary
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COL_PANELS,
	                           BLIS_BUFFER_FOR_B_PANEL );

	// Create control tree objects for packm/unpackm operations on C.
	gemm_packc_cntl
	=
	bli_packm_cntl_obj_create( BLIS_UNBLOCKED,
	                           BLIS_VARIANT1,
	                           gemm_mr,
	                           gemm_nr,
	                           FALSE, // do NOT scale by beta
	                           FALSE, // already dense; densify not necessary
	                           FALSE, // do NOT invert diagonal
	                           FALSE, // reverse iteration if upper?
	                           FALSE, // reverse iteration if lower?
	                           BLIS_PACKED_COLUMNS,
	                           BLIS_BUFFER_FOR_C_PANEL );

	gemm_unpackc_cntl
	=
	bli_unpackm_cntl_obj_create( BLIS_UNBLOCKED,
	                             BLIS_VARIANT1,
	                             NULL ); // no blocksize needed


	//
	// Create a control tree for packing A and B, and streaming C.
	//

	// Create control tree object for lowest-level block-panel kernel.
	gemm_cntl_bp_ke
	=
	bli_gemm_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT2,
	                          NULL, NULL, NULL, NULL,
	                          NULL, NULL, NULL, NULL );

	// Create control tree object for outer panel (to block-panel)
	// problem.
	gemm_cntl_op_bp
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemm_mc,
	                          gemm_ni,
	                          NULL,
	                          gemm_packa_cntl,
	                          gemm_packb_cntl,
	                          NULL,
	                          gemm_cntl_bp_ke,
	                          NULL );

	// Create control tree object for general problem via multiple
	// rank-k (outer panel) updates.
	gemm_cntl_mm_op
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          gemm_kc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm_cntl_op_bp,
	                          NULL );

	// Create control tree object for very large problem via multiple
	// general problems.
	gemm_cntl_vl_mm
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          gemm_nc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm_cntl_mm_op,
	                          NULL );
	
    // Alias the "master" gemm control tree to a shorter name.
	gemm_cntl = gemm_cntl_vl_mm;


	//
	// Create a control tree for packing A, and streaming B and C.
	//

	gemm_cntl_bp_ke5
	=
	bli_gemm_cntl_obj_create( BLIS_UNB_OPT,
	                          BLIS_VARIANT5,
	                          NULL, NULL, NULL, NULL,
	                          NULL, NULL, NULL, NULL );
	gemm_cntl_pm_bp
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT3,
	                          gemm_kc,
	                          NULL,
	                          NULL,
	                          gemm_packa_cntl,
	                          NULL,
	                          //gemm_packc_cntl,
	                          NULL,
	                          gemm_cntl_bp_ke5,
	                          //gemm_unpackc_cntl );
	                          NULL );

	gemm_cntl_mm_pm
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemm_mc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm_cntl_pm_bp,
	                          NULL );

	gemm_cntl_vl_mm5
	=
	bli_gemm_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          gemm_nc,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          NULL,
	                          gemm_cntl_mm_pm,
	                          NULL );

	gemm_cntl_packa = gemm_cntl_vl_mm5;


    if(do_gridlike == 1)
        bli_gemm_grid_cntl_create();
    else
        bli_gemm_hier_cntl_create();
}


void bli_gemm_grid_cntl_create()
{

    gemm_t*           gemm_cntl_bp_ke_mt;
    gemm_t*           gemm_cntl_op_bp_mt;
    gemm_t*           gemm_cntl_mm_op_mt;
    gemm_t*           gemm_cntl_vl_mm_mt;
    packm_t*          gemm_packa_cntl_mt;
    packm_t*          gemm_packb_cntl_mt;

    gemm_num_threads_default = l5_nt*l4_nt*l3_nt*l2_nt*l1_nt;
    dim_t num_l3_comms = l3_nt * l4_nt;
    dim_t threads_per_l3_comm = l5_nt * l2_nt * l1_nt;

    gemm_cntl_mts = (gemm_t**) malloc(sizeof(gemm_t*) * gemm_num_threads_default );
    thread_comm_t*  global_comm = bli_create_communicator( gemm_num_threads_default );

    thread_comm_t** l3_a_comms = (thread_comm_t**) bli_malloc( sizeof( thread_comm_t* ) * num_l3_comms );
    
    for( int i = 0; i < num_l3_comms; i++ ) {
        l3_a_comms[i] = bli_create_communicator( threads_per_l3_comm );
    }

    for( int g = 0; g < l5_nt; g++ )
    {
        thread_comm_t*  l5_comm = bli_create_communicator( l4_nt*l3_nt*l2_nt*l1_nt );
        for( int h = 0; h < l4_nt; h++ )
        {
            thread_comm_t* l4_comm = bli_create_communicator( l3_nt*l2_nt*l1_nt );
            for( int i = 0; i < l3_nt; i++ )
            {
                thread_comm_t* l3_c_comm = bli_create_communicator( l2_nt*l1_nt );
                for( int j = 0; j < l2_nt; j++ )
                {
                    for( int k = 0; k < l1_nt; k++ )
                    {
                        //TODO: doublecheck this
                        dim_t l3_c_comm_id = j * l1_nt + k;
                        dim_t l3_a_comm_id = l3_c_comm_id + l2_nt * g;
                        thread_comm_t* l3_a_comm = l3_a_comms[ i + h * l3_nt ];
                        dim_t l4_comm_id = i * l2_nt + j * l1_nt + k;
                        dim_t l5_comm_id = h * l4_comm->num_threads + l4_comm_id;
                        dim_t global_comm_id = g * l5_comm->num_threads + l5_comm_id; 

                        //Next two lines don't work in the general case.

                        packm_thread_info_t* pack_a_info = bli_create_packm_thread_info( l3_a_comm, l3_a_comm_id, max_pack_with );
                        gemm_packa_cntl_mt = bli_packm_cntl_obj_create_mt( BLIS_BLOCKED, BLIS_VARIANT2, gemm_mr,
                                       gemm_kr, FALSE, FALSE, FALSE, FALSE, FALSE,
                                       BLIS_PACKED_ROW_PANELS, BLIS_BUFFER_FOR_A_BLOCK, pack_a_info );

                        packm_thread_info_t* pack_b_info = bli_create_packm_thread_info( l4_comm, l4_comm_id, max_pack_with );
                        gemm_packb_cntl_mt = bli_packm_cntl_obj_create_mt( BLIS_BLOCKED, BLIS_VARIANT2,
                                       gemm_kr,  gemm_nr, 
                                       FALSE, FALSE, FALSE, FALSE, FALSE, 
                                       BLIS_PACKED_COL_PANELS, BLIS_BUFFER_FOR_B_PANEL, pack_b_info );

                        gemm_ker_thread_info_t* l2_info = bli_create_gemm_ker_thread_info( j, l2_nt, k, l1_nt, 0 );
                        gemm_cntl_bp_ke_mt = bli_gemm_cntl_obj_create_mt( BLIS_UNB_OPT, BLIS_VARIANT2, NULL, NULL, NULL, NULL, 
                                NULL, NULL, NULL, NULL, l2_info );

                        gemm_blk_thread_info_t* l3_info = bli_create_gemm_blk_thread_info( l3_a_comm, l3_a_comm_id, l4_comm, l4_comm_id, l3_c_comm, j, l3_nt, i );
                        gemm_cntl_op_bp_mt = bli_gemm_cntl_obj_create_mt( BLIS_BLOCKED, BLIS_VARIANT1, gemm_mc, gemm_ni, 
                                NULL, gemm_packa_cntl_mt, gemm_packb_cntl_mt, NULL, gemm_cntl_bp_ke_mt, NULL, l3_info );
                                                  
                        gemm_blk_thread_info_t* l4_info = bli_create_gemm_blk_thread_info(  l4_comm, l4_comm_id, l4_comm, l4_comm_id, l5_comm, l5_comm_id, l4_nt, h );
                        gemm_cntl_mm_op_mt = bli_gemm_cntl_obj_create_mt( BLIS_BLOCKED,BLIS_VARIANT3, gemm_kc,
                                NULL, NULL, NULL, NULL, NULL, gemm_cntl_op_bp_mt, NULL, l4_info );

                        gemm_blk_thread_info_t* l5_info = bli_create_gemm_blk_thread_info( global_comm, global_comm_id, l5_comm, l5_comm_id, l5_comm, l5_comm_id, l5_nt, g );
                        gemm_cntl_vl_mm_mt = bli_gemm_cntl_obj_create_mt( BLIS_BLOCKED, BLIS_VARIANT2, gemm_nc, NULL, NULL,NULL,NULL,NULL,
                                  gemm_cntl_mm_op_mt, NULL, l5_info );
                                  
                        gemm_cntl_mts[global_comm_id] = gemm_cntl_vl_mm_mt;
                    }
                }
            }
        }
    }
}

void bli_gemm_hier_cntl_create()
{
    gemm_t*           gemm_cntl_bp_ke_mt;
    gemm_t*           gemm_cntl_op_bp_mt;
    gemm_t*           gemm_cntl_mm_op_mt;
    gemm_t*           gemm_cntl_vl_mm_mt;
    packm_t*          gemm_packa_cntl_mt;
    packm_t*          gemm_packb_cntl_mt;
    
    char * str = getenv( "BLIS_L0_NT" );
    if(str != NULL )
        l0_nt = strtol( str, NULL, 10 ); 
    str = getenv( "BLIS_L1_NT" );
    if(str != NULL )
        l1_nt = strtol( str, NULL, 10 ); 
    str = getenv( "BLIS_L2_NT" );
    if(str != NULL )
        l2_nt = strtol( str, NULL, 10 ); 
    str = getenv( "BLIS_L3_NT" );
    if(str != NULL )
        l3_nt = strtol( str, NULL, 10 ); 
    str = getenv( "BLIS_L4_NT" );
    if(str != NULL )
        l4_nt = strtol( str, NULL, 10 ); 
    str = getenv( "BLIS_L5_NT" );
    if(str != NULL )
        l5_nt = strtol( str, NULL, 10 ); 

    gemm_num_threads_default = l5_nt*l4_nt*l3_nt*l2_nt*l1_nt*l0_nt;
    gemm_cntl_mts = (gemm_t**) bli_malloc(sizeof(gemm_t) * gemm_num_threads_default );

    thread_comm_t*  global_comm = bli_create_communicator( gemm_num_threads_default );
    for( int g = 0; g < l5_nt; g++ )
    {
        thread_comm_t*  l5_comm = bli_create_communicator( l4_nt*l3_nt*l2_nt*l1_nt*l0_nt );
        for( int h = 0; h < l4_nt; h++ )
        {
            thread_comm_t* l4_comm = bli_create_communicator( l3_nt*l2_nt*l1_nt*l0_nt );
            for( int i = 0; i < l3_nt; i++ )
            {
                thread_comm_t* l3_comm = bli_create_communicator( l2_nt*l1_nt*l0_nt );
                for( int j = 0; j < l2_nt; j++ )
                {
                    for( int k = 0; k < l1_nt; k++) 
                    {
                        for(int m = 0; m < l0_nt; m++)
                        {
                        //TODO: doublecheck this
                        dim_t l3_comm_id = m + k * l0_nt + j * l1_nt * l0_nt;
                        dim_t l4_comm_id = i * l3_comm->num_threads + l3_comm_id;
                        dim_t l5_comm_id = h * l4_comm->num_threads + l4_comm_id;
                        dim_t global_comm_id = g * l5_comm->num_threads + l5_comm_id; 
                        //printf("%d\t%d\t%d\t%d\t%d\n", l3_comm_id, l4_comm_id, global_comm_id, m, k);

                        packm_thread_info_t* pack_a_info = bli_create_packm_thread_info( l3_comm, l3_comm_id, max_pack_with );
                        gemm_packa_cntl_mt = bli_packm_cntl_obj_create_mt( BLIS_BLOCKED, BLIS_VARIANT2, gemm_mr, 
                                       gemm_kr,  FALSE, FALSE, FALSE, FALSE, FALSE,
                                       BLIS_PACKED_ROW_PANELS, BLIS_BUFFER_FOR_A_BLOCK, pack_a_info );


                        packm_thread_info_t* pack_b_info = bli_create_packm_thread_info(  l4_comm, l4_comm_id, max_pack_with );
                        gemm_packb_cntl_mt = bli_packm_cntl_obj_create_mt( BLIS_BLOCKED, BLIS_VARIANT2,
                                       gemm_kr,  gemm_nr, 
                                       FALSE, FALSE, FALSE, FALSE, FALSE, 
                                       BLIS_PACKED_COL_PANELS, BLIS_BUFFER_FOR_B_PANEL, pack_b_info );

                        gemm_ker_thread_info_t* l2_info = bli_create_gemm_ker_thread_info( j, l2_nt, k, l1_nt, m );
                        gemm_cntl_bp_ke_mt = bli_gemm_cntl_obj_create_mt( BLIS_UNB_OPT, BLIS_VARIANT2, NULL, NULL, NULL, NULL, 
                                NULL, NULL, NULL, NULL, l2_info );

                        gemm_blk_thread_info_t* l3_info = bli_create_gemm_blk_thread_info( l3_comm, l3_comm_id, l4_comm, l4_comm_id, l3_comm, l3_comm_id, l3_nt, i );
                        gemm_cntl_op_bp_mt = bli_gemm_cntl_obj_create_mt( BLIS_BLOCKED, BLIS_VARIANT1, gemm_mc, gemm_ni, 
                                NULL, gemm_packa_cntl_mt, gemm_packb_cntl_mt, NULL, gemm_cntl_bp_ke_mt, NULL, l3_info );
                                                  
                        gemm_blk_thread_info_t* l4_info = bli_create_gemm_blk_thread_info( l4_comm, l4_comm_id, l4_comm, l4_comm_id, l5_comm, l5_comm_id, l4_nt, h );
                        gemm_cntl_mm_op_mt = bli_gemm_cntl_obj_create_mt( BLIS_BLOCKED,BLIS_VARIANT3, gemm_kc,
                                NULL, NULL, NULL, NULL, NULL, gemm_cntl_op_bp_mt, NULL, l4_info );

                        gemm_blk_thread_info_t* l5_info = bli_create_gemm_blk_thread_info( global_comm, global_comm_id, l5_comm, l5_comm_id, l5_comm, l5_comm_id, l5_nt, g );
                        gemm_cntl_vl_mm_mt = bli_gemm_cntl_obj_create_mt( BLIS_BLOCKED, BLIS_VARIANT2, gemm_nc, NULL, NULL,NULL,NULL,NULL,
                                  gemm_cntl_mm_op_mt, NULL, l5_info );
                        
                        gemm_cntl_mts[global_comm_id] = gemm_cntl_vl_mm_mt;
                        }
                    }
                }
            }
        }
    }
}

// Stuff to free at the end
// Need to rewrite this to support freeing arbitrary cntl trees
void bli_free_gemm_multithreaded_cntls()
{
    for( int i = 0; i < gemm_num_threads_default; i++)
    {
        gemm_t* gemm_cntl_vl_mm_mt = gemm_cntl_mts[i];
        gemm_t* gemm_cntl_mm_op_mt = gemm_cntl_vl_mm_mt->sub_gemm;
        gemm_t* gemm_cntl_op_bp_mt = gemm_cntl_mm_op_mt->sub_gemm;
        gemm_t* gemm_cntl_bp_ke_mt = gemm_cntl_op_bp_mt->sub_gemm;
        
        //free infos
        bli_gemm_ker_thread_info_free( gemm_cntl_bp_ke_mt->thread_info );
        bli_gemm_blk_thread_info_free( gemm_cntl_op_bp_mt->thread_info );
        bli_gemm_blk_thread_info_free( gemm_cntl_mm_op_mt->thread_info );
        bli_gemm_blk_thread_info_free( gemm_cntl_vl_mm_mt->thread_info );

        //Free packm cntls
        packm_t* packa_mt = gemm_cntl_op_bp->sub_packm_a;
        bli_packm_thread_info_free( packa_mt->thread_info );

        packm_t* packb_mt = gemm_cntl_op_bp->sub_packm_b;
        bli_packm_thread_info_free( packb_mt->thread_info );

        bli_cntl_obj_free( packa_mt );
        bli_cntl_obj_free( packb_mt );
        bli_cntl_obj_free( gemm_cntl_vl_mm_mt );
        bli_cntl_obj_free( gemm_cntl_mm_op_mt );
        bli_cntl_obj_free( gemm_cntl_op_bp_mt );
        bli_cntl_obj_free( gemm_cntl_bp_ke_mt );
    }
}

void bli_gemm_cntl_finalize()
{
    //bli_free_gemm_multithreaded_cntls();

	bli_blksz_obj_free( gemm_mc );
	bli_blksz_obj_free( gemm_nc );
	bli_blksz_obj_free( gemm_kc );
	bli_blksz_obj_free( gemm_mr );
	bli_blksz_obj_free( gemm_nr );
	bli_blksz_obj_free( gemm_kr );
	bli_blksz_obj_free( gemm_ni );

	bli_cntl_obj_free( gemm_packa_cntl );
	bli_cntl_obj_free( gemm_packb_cntl );
	bli_cntl_obj_free( gemm_packc_cntl );
	bli_cntl_obj_free( gemm_unpackc_cntl );

	bli_cntl_obj_free( gemm_cntl_bp_ke );
	bli_cntl_obj_free( gemm_cntl_op_bp );
	bli_cntl_obj_free( gemm_cntl_mm_op );
	bli_cntl_obj_free( gemm_cntl_vl_mm );

	bli_cntl_obj_free( gemm_cntl_bp_ke5 );
	bli_cntl_obj_free( gemm_cntl_pm_bp );
	bli_cntl_obj_free( gemm_cntl_mm_pm );
	bli_cntl_obj_free( gemm_cntl_vl_mm5 );
}

gemm_t* bli_gemm_cntl_obj_create( impl_t     impl_type,
                                  varnum_t   var_num,
                                  blksz_t*   b,
                                  blksz_t*   b_aux,
                                  scalm_t*   sub_scalm,
                                  packm_t*   sub_packm_a,
                                  packm_t*   sub_packm_b,
                                  packm_t*   sub_packm_c,
                                  gemm_t*    sub_gemm,
                                  unpackm_t* sub_unpackm_c )
{
    return bli_gemm_cntl_obj_create_mt( impl_type, var_num, b, b_aux, sub_scalm, sub_packm_a,
        sub_packm_b, sub_packm_c, sub_gemm, sub_unpackm_c, NULL );
}

gemm_t* bli_gemm_cntl_obj_create_mt( impl_t     impl_type,
                                  varnum_t   var_num,
                                  blksz_t*   b,
                                  blksz_t*   b_aux,
                                  scalm_t*   sub_scalm,
                                  packm_t*   sub_packm_a,
                                  packm_t*   sub_packm_b,
                                  packm_t*   sub_packm_c,
                                  gemm_t*    sub_gemm,
                                  unpackm_t* sub_unpackm_c,
                                  void*      thread_info )
{
	gemm_t* cntl;

	cntl = ( gemm_t* ) bli_malloc( sizeof(gemm_t) );	

	cntl->impl_type     = impl_type;
	cntl->var_num       = var_num;
	cntl->b             = b;
	cntl->b_aux         = b_aux;
	cntl->sub_scalm     = sub_scalm;
	cntl->sub_packm_a   = sub_packm_a;
	cntl->sub_packm_b   = sub_packm_b;
	cntl->sub_packm_c   = sub_packm_c;
	cntl->sub_gemm      = sub_gemm;
	cntl->sub_unpackm_c = sub_unpackm_c;
    cntl->thread_info   = thread_info;

	return cntl;
}

