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

packm_t* packm_cntl_row_noscale;
packm_t* packm_cntl_row_scale;
packm_t* packm_cntl_col_noscale;
packm_t* packm_cntl_col_scale;

packm_t* packm_cntl_rpn_noscale;
packm_t* packm_cntl_rpn_scale;
packm_t* packm_cntl_cpn_noscale;
packm_t* packm_cntl_cpn_scale;

packm_t* packm_cntl_noscale;
packm_t* packm_cntl_scale;

blksz_t* packm_mult_ldim;
blksz_t* packm_mult_nvec;

void bli_packm_cntl_init()
{
	// Create blocksize objects for m and n register blocking. We will attach
	// these to the packm control node so they can be used to (a) allocate a
	// block whose m and n dimension are multiples of mr and nr, and (b) know
	// how much zero-padding is necessary for edge cases. 
	// NOTE: these alignments end up getting applied to matrices packed for
	// level-2 operations, even though they are not needed, and/or smaller
	// alignments may be sufficient. For simplicity, we choose to tweak the
	// dimensions of all pack matrix buffers the same amount.
	packm_mult_ldim
	=
	bli_blksz_obj_create( BLIS_DEFAULT_MR_S, 0,
	                      BLIS_DEFAULT_MR_D, 0,
	                      BLIS_DEFAULT_MR_C, 0,
	                      BLIS_DEFAULT_MR_Z, 0 );

	packm_mult_nvec
	=
	bli_blksz_obj_create( BLIS_DEFAULT_NR_S, 0,
	                      BLIS_DEFAULT_NR_D, 0,
	                      BLIS_DEFAULT_NR_C, 0,
	                      BLIS_DEFAULT_NR_Z, 0 );

	// Generally speaking, the BLIS_PACKED_ROWS and BLIS_PACKED_COLUMNS
	// are used by the level-2 operations, and thus densification is not
	// necessary. These schemas amount to simple copies to row or column
	// storage. These simple schemas may be used by level-3 operations,
	// but they should never be used for matrices with structure (since
	// they do not densify).
	// The BLIS_PACKED_ROW_PANELS and BLIS_PACKED_COL_PANELS schemas are
	// used only in level-3 operations. They pack to (typically) skinny
	// row and column panels, where the width of the panel is determined
	// by register blocksizes. They are configured to densify matrices
	// with structure, though they can also be used on matrices that
	// are already dense and/or have no structure.

	// Create control trees to pack by rows (with and without scaling).
	packm_cntl_row_noscale
	=
	bli_packm_cntl_obj_create( BLIS_UNBLOCKED,
	                           BLIS_VARIANT1,    // When packing to rows:
	                           packm_mult_nvec,  // - nvec multiple is used for m dimension
	                           packm_mult_ldim,  // - ldim multiple is used for n dimension
	                           FALSE,            // do NOT scale
	                           FALSE,            // do NOT densify structure
	                           FALSE,            // do NOT invert diagonal
	                           FALSE,            // do NOT iterate backwards if upper
	                           FALSE,            // do NOT iterate backwards if lower
	                           BLIS_PACKED_ROWS,
	                           BLIS_BUFFER_FOR_GEN_USE );
	packm_cntl_row_scale
	=
	bli_packm_cntl_obj_create( BLIS_UNBLOCKED,
	                           BLIS_VARIANT1,    // When packing to rows:
	                           packm_mult_nvec,  // - nvec multiple is used for m dimension
	                           packm_mult_ldim,  // - ldim multiple is used for n dimension
	                           TRUE,             // do scale
	                           FALSE,            // do NOT densify structure
	                           FALSE,            // do NOT invert diagonal
	                           FALSE,            // do NOT iterate backwards if upper
	                           FALSE,            // do NOT iterate backwards if lower
	                           BLIS_PACKED_ROWS,
	                           BLIS_BUFFER_FOR_GEN_USE );


	// Create control trees to pack by columns (with and without scaling).
	packm_cntl_col_noscale
	=
	bli_packm_cntl_obj_create( BLIS_UNBLOCKED,
	                           BLIS_VARIANT1,    // When packing to columns:
	                           packm_mult_ldim,  // - ldim multiple is used for m dimension
	                           packm_mult_nvec,  // - nvec multiple is used for n dimension
	                           FALSE,            // do NOT scale
	                           FALSE,            // do NOT densify structure
	                           FALSE,            // do NOT invert diagonal
	                           FALSE,            // do NOT iterate backwards if upper
	                           FALSE,            // do NOT iterate backwards if lower
	                           BLIS_PACKED_COLUMNS,
	                           BLIS_BUFFER_FOR_GEN_USE );
	packm_cntl_col_scale
	=
	bli_packm_cntl_obj_create( BLIS_UNBLOCKED,
	                           BLIS_VARIANT1,    // When packing to columns:
	                           packm_mult_ldim,  // - ldim multiple is used for m dimension
	                           packm_mult_nvec,  // - nvec multiple is used for n dimension
	                           TRUE,             // do scale
	                           FALSE,            // do NOT densify structure
	                           FALSE,            // do NOT invert diagonal
	                           FALSE,            // do NOT iterate backwards if upper
	                           FALSE,            // do NOT iterate backwards if lower
	                           BLIS_PACKED_COLUMNS,
	                           BLIS_BUFFER_FOR_GEN_USE );


	// Set defaults when we don't care whether the packing is by rows or
	// by columns.
	packm_cntl_noscale = packm_cntl_col_noscale;
	packm_cntl_scale   = packm_cntl_col_scale;
}

void bli_packm_cntl_finalize()
{
	bli_cntl_obj_free( packm_cntl_row_noscale );
	bli_cntl_obj_free( packm_cntl_row_scale );
	bli_cntl_obj_free( packm_cntl_col_noscale );
	bli_cntl_obj_free( packm_cntl_col_scale );

	bli_blksz_obj_free( packm_mult_ldim );
	bli_blksz_obj_free( packm_mult_nvec );
}

packm_t* bli_packm_cntl_obj_create_mt( impl_t     impl_type,
                                    varnum_t   var_num,
                                    blksz_t*   mr,
                                    blksz_t*   nr,
                                    bool_t     does_scale,
                                    bool_t     does_densify,
                                    bool_t     does_invert_diag,
                                    bool_t     rev_iter_if_upper,
                                    bool_t     rev_iter_if_lower,
                                    pack_t     pack_schema,
                                    packbuf_t  pack_buf_type,
                                    packm_thread_info_t* thread_info )
{
	packm_t* cntl;

	cntl = ( packm_t* ) bli_malloc( sizeof(packm_t) );

	cntl->impl_type         = impl_type;
	cntl->var_num           = var_num;
	cntl->mr                = mr;
	cntl->nr                = nr;
	cntl->does_scale        = does_scale;
	cntl->does_densify      = does_densify;
	cntl->does_invert_diag  = does_invert_diag;
	cntl->rev_iter_if_upper = rev_iter_if_upper;
	cntl->rev_iter_if_lower = rev_iter_if_lower;
	cntl->pack_schema       = pack_schema;
	cntl->pack_buf_type     = pack_buf_type;
    cntl->thread_info       = thread_info;

	return cntl;
}

void bli_packm_cntl_obj_init_mt( packm_t*   cntl,
                              impl_t     impl_type,
                              varnum_t   var_num,
                              blksz_t*   mr,
                              blksz_t*   nr,
                              bool_t     does_scale,
                              bool_t     does_densify,
                              bool_t     does_invert_diag,
                              bool_t     rev_iter_if_upper,
                              bool_t     rev_iter_if_lower,
                              pack_t     pack_schema,
                              packbuf_t  pack_buf_type,
                              packm_thread_info_t* thread_info )
{
	cntl->impl_type         = impl_type;
	cntl->var_num           = var_num;
	cntl->mr            = mr;
	cntl->nr            = nr;
	cntl->does_scale        = does_scale;
	cntl->does_densify      = does_densify;
	cntl->does_invert_diag  = does_invert_diag;
	cntl->rev_iter_if_upper = rev_iter_if_upper;
	cntl->rev_iter_if_lower = rev_iter_if_lower;
	cntl->pack_schema       = pack_schema;
	cntl->pack_buf_type     = pack_buf_type;
    cntl->thread_info       = thread_info;
}

packm_t* bli_packm_cntl_obj_create( impl_t     impl_type,
                                    varnum_t   var_num,
                                    blksz_t*   mr,
                                    blksz_t*   nr,
                                    bool_t     does_scale,
                                    bool_t     does_densify,
                                    bool_t     does_invert_diag,
                                    bool_t     rev_iter_if_upper,
                                    bool_t     rev_iter_if_lower,
                                    pack_t     pack_schema,
                                    packbuf_t  pack_buf_type )
{
    return bli_packm_cntl_obj_create_mt( impl_type, var_num, mr, nr,
        does_scale, does_densify, does_invert_diag, rev_iter_if_upper, rev_iter_if_lower,
        pack_schema, pack_buf_type, NULL);
}

void bli_packm_cntl_obj_init( packm_t*   cntl,
                              impl_t     impl_type,
                              varnum_t   var_num,
                              blksz_t*   mr,
                              blksz_t*   nr,
                              bool_t     does_scale,
                              bool_t     does_densify,
                              bool_t     does_invert_diag,
                              bool_t     rev_iter_if_upper,
                              bool_t     rev_iter_if_lower,
                              pack_t     pack_schema,
                              packbuf_t  pack_buf_type )
{
    bli_packm_cntl_obj_init_mt( cntl, impl_type, var_num, mr, nr,
        does_scale, does_densify, does_invert_diag, rev_iter_if_upper, rev_iter_if_lower,
        pack_schema, pack_buf_type, NULL);
}

