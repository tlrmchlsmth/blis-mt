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

struct packm_s
{
	impl_t         impl_type;
	varnum_t       var_num;
	blksz_t*       mr;
	blksz_t*       nr;
	bool_t         does_scale;
	bool_t         does_densify;
	bool_t         does_invert_diag;
	bool_t         rev_iter_if_upper;
	bool_t         rev_iter_if_lower;
	pack_t         pack_schema;
	packbuf_t      pack_buf_type;
    packm_thread_info_t* thread_info;
};
typedef struct packm_s packm_t;

#define cntl_mr( cntl )                cntl->mr
#define cntl_nr( cntl )                cntl->nr

#define cntl_does_scale( cntl )        cntl->does_scale
#define cntl_does_densify( cntl )      cntl->does_densify
#define cntl_does_invert_diag( cntl )  cntl->does_invert_diag
#define cntl_rev_iter_if_upper( cntl ) cntl->rev_iter_if_upper
#define cntl_rev_iter_if_lower( cntl ) cntl->rev_iter_if_lower
#define cntl_pack_schema( cntl )       cntl->pack_schema
#define cntl_pack_buf_type( cntl )     cntl->pack_buf_type

#define cntl_sub_packm( cntl )         cntl->sub_packm
#define cntl_sub_packm_a( cntl )       cntl->sub_packm_a
#define cntl_sub_packm_a11( cntl )     cntl->sub_packm_a11
#define cntl_sub_packm_b( cntl )       cntl->sub_packm_b
#define cntl_sub_packm_b11( cntl )     cntl->sub_packm_b11
#define cntl_sub_packm_c( cntl )       cntl->sub_packm_c
#define cntl_sub_packm_c11( cntl )     cntl->sub_packm_c11

void     bli_packm_cntl_init( void );
void     bli_packm_cntl_finalize( void );
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
                                    packm_thread_info_t* thread_info );

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
                              packm_thread_info_t* thread_info );

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
                                    packbuf_t  pack_buf_type );

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
                              packbuf_t  pack_buf_type );

