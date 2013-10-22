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
#include "test_libblis.h"


// Static variables.
static char*     op_str                    = "trsm_ukr";
static char*     o_types                   = "m";  // c
static char*     p_types                   = "u";  // uploa
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_trsm_ukr_deps( test_params_t* params,
                                 test_op_t*     op );

void libblis_test_trsm_ukr_experiment( test_params_t* params,
                                       test_op_t*     op,
                                       mt_impl_t      impl,
                                       num_t          datatype,
                                       char*          pc_str,
                                       char*          sc_str,
                                       unsigned int   p_cur,
                                       double*        perf,
                                       double*        resid );

void libblis_test_trsm_ukr_impl( mt_impl_t impl,
                                 side_t    side,
                                 obj_t*    a,
                                 obj_t*    b,
                                 obj_t*    bd,
                                 obj_t*    c );

void libblis_test_trsm_ukr_check( side_t  side,
                                  obj_t*  a,
                                  obj_t*  b,
                                  obj_t*  b_orig,
                                  double* resid );



void libblis_test_trsm_ukr_deps( test_params_t* params, test_op_t* op )
{
	libblis_test_randv( params, &(op->ops->randv) );
	libblis_test_randm( params, &(op->ops->randm) );
	libblis_test_setv( params, &(op->ops->setv) );
	libblis_test_fnormv( params, &(op->ops->fnormv) );
	libblis_test_subv( params, &(op->ops->subv) );
	libblis_test_scalv( params, &(op->ops->scalv) );
	libblis_test_copym( params, &(op->ops->copym) );
	libblis_test_scalm( params, &(op->ops->scalm) );
	libblis_test_gemv( params, &(op->ops->gemv) );
	libblis_test_trsv( params, &(op->ops->trsv) );
}



void libblis_test_trsm_ukr( test_params_t* params, test_op_t* op )
{

	// Return early if this test has already been done.
	if ( op->test_done == TRUE ) return;

	// Return early if operation is disabled.
	if ( op->op_switch == DISABLE_ALL ||
	     op->ops->l3ukr_over == DISABLE_ALL ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_trsm_ukr_deps( params, op );

	// Execute the test driver for each implementation requested.
	if ( op->front_seq == ENABLE )
	{
		libblis_test_op_driver( params,
		                        op,
		                        BLIS_TEST_SEQ_UKERNEL,
		                        op_str,
		                        p_types,
		                        o_types,
		                        thresh,
		                        libblis_test_trsm_ukr_experiment );
	}
}


// Import the register blocksizes used by the micro-kernel(s).
extern blksz_t* gemm_mr;
extern blksz_t* gemm_nr;
extern blksz_t* gemm_kr;

void libblis_test_trsm_ukr_experiment( test_params_t* params,
                                       test_op_t*     op,
                                       mt_impl_t      impl,
                                       num_t          datatype,
                                       char*          pc_str,
                                       char*          sc_str,
                                       unsigned int   p_cur,
                                       double*        perf,
                                       double*        resid )
{
	unsigned int n_repeats = params->n_repeats;
	unsigned int i;

	double       time_min  = 1e9;
	double       time;

	dim_t        m, n;

	char         sc_a = 'c';
	char         sc_b = 'r';

	side_t       side = BLIS_LEFT;
	uplo_t       uploa;

	obj_t        kappa;
	obj_t        a, b, bd, c;
	obj_t        ap, bp;
	obj_t        c_save;


	// Fix m and n to MR and NR, respectively.
	m = bli_blksz_for_type( datatype, gemm_mr );
	n = bli_blksz_for_type( datatype, gemm_nr );

	// Store the register blocksizes so that the driver can retrieve the
	// values later when printing results.
	op->dim_aux[0] = m;
	op->dim_aux[1] = n;

	// Map parameter characters to BLIS constants.
	bli_param_map_char_to_blis_uplo( pc_str[0], &uploa );

	// Create test scalars.
	bli_obj_init_scalar( datatype, &kappa );

	// Create test operands (vectors and/or matrices).
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_a,      m, m, &a );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_b,      m, n, &b );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m, n, &c );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m, n, &c_save );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_b,      m, 4*n, &bd );

	// Set the structure, uplo, and diagonal offset properties of A.
	bli_obj_set_struc( BLIS_TRIANGULAR, a );
	bli_obj_set_uplo( uploa, a );
	bli_obj_set_diag_offset( 0, a );

	// Randomize A, make it densely triangular.
	bli_randm( &a );
	bli_mktrim( &a );

	// Randomize B.
	bli_randm( &b );

	// Randomize C and save C.
	bli_randm( &c );
	bli_copym( &c, &c_save );

	// Normalize by m.
	bli_setsc( 1.0/( double )m, 0.0, &kappa );
	bli_scalm( &kappa, &b );


	// Initialize pack objects.
	bli_obj_init_pack( &ap );
	bli_obj_init_pack( &bp );

	// Create pack objects for a and b.
	libblis_test_pobj_create( gemm_mr,
	                          gemm_mr,
	                          BLIS_INVERT_DIAG,
	                          BLIS_PACKED_ROW_PANELS,
	                          BLIS_BUFFER_FOR_A_BLOCK,
	                          &a, &ap );
	libblis_test_pobj_create( gemm_mr,
	                          gemm_nr,
	                          BLIS_NO_INVERT_DIAG,
	                          BLIS_PACKED_COL_PANELS,
	                          BLIS_BUFFER_FOR_B_PANEL,
	                          &b, &bp );

	// Pack the contents of a to ap.
	bli_packm_blk_var3( &BLIS_ONE, &a, &ap );


	// Repeat the experiment n_repeats times and record results. 
	for ( i = 0; i < n_repeats; ++i )
	{
		// Re-pack the contents of b to bp.
		bli_packm_blk_var2( &BLIS_ONE, &b, &bp );

		// Re-duplicate the contents of bp to bd.
		bli_dupl( &bp, &bd );

		bli_copym( &c_save, &c );

		time = bli_clock();

		libblis_test_trsm_ukr_impl( impl, side, &ap, &bp, &bd, &c );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 1.0 * m * m * n ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( b ) ) *perf *= 4.0;

	// Perform checks.
	libblis_test_trsm_ukr_check( side, &a, &c, &b, resid );

	// Zero out performance and residual if output matrix is empty.
	libblis_test_check_empty_problem( &c, perf, resid );

	// Release packing buffers within pack objects.
	bli_obj_release_pack( &ap );
	bli_obj_release_pack( &bp );

	// Free the test objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );
	bli_obj_free( &c_save );
}



void libblis_test_trsm_ukr_impl( mt_impl_t impl,
                                 side_t    side,
                                 obj_t*    a,
                                 obj_t*    b,
                                 obj_t*    bd,
                                 obj_t*    c )
{
	switch ( impl )
	{
		case BLIS_TEST_SEQ_UKERNEL:
		bli_trsm_ukr( a, b, bd, c );
		break;

		default:
		libblis_test_printf_error( "Invalid implementation type.\n" );
	}
}



void libblis_test_trsm_ukr_check( side_t  side,
                                  obj_t*  a,
                                  obj_t*  b,
                                  obj_t*  b_orig,
                                  double* resid )
{
	num_t  dt      = bli_obj_datatype( *b );
	num_t  dt_real = bli_obj_datatype_proj_to_real( *b );

	dim_t  m       = bli_obj_length( *b );
	dim_t  n       = bli_obj_width( *b );

	obj_t  kappa, norm;
	obj_t  t, v, w, z;

	double junk;

	//
	// Pre-conditions:
	// - a is randomized and triangular.
	// - b_orig is randomized.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   B := inv(transa(A)) * B_orig    (side = left)
	//   B := B_orig * inv(transa(A))    (side = right)
	//
	// is functioning correctly if
	//
	//   fnorm( v - z )
	//
	// is negligible, where
	//
	//   v = B * t
	//
	//   z = ( inv(transa(A)) * B ) * t     (side = left)
	//     = inv(transa(A)) * B * t
	//     = inv(transa(A)) * w
	//
	//   z = ( B * inv(transa(A)) ) * t     (side = right)
	//     = B * tinv(ransa(A)) * t
	//     = B * w

	bli_obj_init_scalar( dt,      &kappa );
	bli_obj_init_scalar( dt_real, &norm );

	if ( bli_is_left( side ) )
	{
		bli_obj_create( dt, n, 1, 0, 0, &t );
		bli_obj_create( dt, m, 1, 0, 0, &v );
		bli_obj_create( dt, m, 1, 0, 0, &w );
		bli_obj_create( dt, m, 1, 0, 0, &z );
	}
	else // else if ( bli_is_left( side ) )
	{
		bli_obj_create( dt, n, 1, 0, 0, &t );
		bli_obj_create( dt, m, 1, 0, 0, &v );
		bli_obj_create( dt, n, 1, 0, 0, &w );
		bli_obj_create( dt, m, 1, 0, 0, &z );
	}

	bli_randv( &t );
	bli_setsc( 1.0/( double )n, 0.0, &kappa );
	bli_scalv( &kappa, &t );

	bli_gemv( &BLIS_ONE, b, &t, &BLIS_ZERO, &v );

	if ( bli_is_left( side ) )
	{
		bli_gemv( &BLIS_ONE, b_orig, &t, &BLIS_ZERO, &w );
		bli_trsv( &BLIS_ONE, a, &w );
		bli_copyv( &w, &z );
	}
	else
	{
		bli_copyv( &t, &w );
		bli_trsv( &BLIS_ONE, a, &w );
		bli_gemv( &BLIS_ONE, b_orig, &w, &BLIS_ZERO, &z );
	}

	bli_subv( &z, &v );
	bli_fnormv( &v, &norm );
	bli_getsc( &norm, resid, &junk );

	bli_obj_free( &t );
	bli_obj_free( &v );
	bli_obj_free( &w );
	bli_obj_free( &z );
}



//
// Define object-wrapper to TRSM_L_UKERNEL, TRSM_U_UKERNEL micro-kernels.
//

#undef  FUNCPTR_T
#define FUNCPTR_T trsm_ukr_fp

typedef void (*FUNCPTR_T)(
                           void*   a,
                           void*   b,
                           void*   bd,
                           void*   c, inc_t rs_c, inc_t cs_c
                         );

static FUNCPTR_T GENARRAY(ftypes_l,trsm_l_ukr);
static FUNCPTR_T GENARRAY(ftypes_u,trsm_u_ukr);


void bli_trsm_ukr( obj_t*  a,
                   obj_t*  b,
                   obj_t*  bd,
                   obj_t*  c )
{
    num_t     dt        = bli_obj_datatype( *c );

    void*     buf_a     = bli_obj_buffer_at_off( *a );

    void*     buf_b     = bli_obj_buffer_at_off( *b );

    void*     buf_bd    = bli_obj_buffer_at_off( *bd );

    void*     buf_c     = bli_obj_buffer_at_off( *c );
    inc_t     rs_c      = bli_obj_row_stride( *c );
    inc_t     cs_c      = bli_obj_col_stride( *c );

    FUNCPTR_T f;

    // Index into the type combination array to extract the correct
    // function pointer.
    if ( bli_obj_is_lower( *a ) ) f = ftypes_l[dt];
    else                          f = ftypes_u[dt];

    // Invoke the function.
    f( buf_a,
       buf_b,
       buf_bd,
       buf_c, rs_c, cs_c );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, ukrname ) \
\
void PASTEMAC(ch,varname)( \
                           void*   a, \
                           void*   b, \
                           void*   bd, \
                           void*   c, inc_t rs_c, inc_t cs_c \
                         ) \
{ \
    PASTEMAC(ch,ukrname)( a, \
                          b, \
                          bd, \
                          c, rs_c, cs_c ); \
}

INSERT_GENTFUNC_BASIC( trsm_l_ukr, TRSM_L_UKERNEL )
INSERT_GENTFUNC_BASIC( trsm_u_ukr, TRSM_U_UKERNEL )

