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

#define FUNCPTR_T trmm_fp

typedef void (*FUNCPTR_T)( obj_t*  alpha,
                           obj_t*  a,
                           obj_t*  b,
                           obj_t*  beta,
                           obj_t*  c,
                           trmm_t* cntl );

<<<<<<< HEAD
static FUNCPTR_T vars[2][2][4][3] =
{
	// left
	{
		// lower
		{
		    // unblocked            optimized unblocked    blocked
		    { NULL,                 NULL,                  bli_trmm_blk_var1   },
		    { NULL,                 bli_trmm_ll_ker_var2,  bli_trmm_blk_var2b  },
		    { NULL,                 NULL,                  bli_trmm_blk_var3b  },
		    { NULL,                 NULL,                  NULL                },
		},
		// upper
		{
		    // unblocked            optimized unblocked    blocked
		    { NULL,                 NULL,                  bli_trmm_blk_var1   },
		    { NULL,                 bli_trmm_lu_ker_var2,  bli_trmm_blk_var2f  },
		    { NULL,                 NULL,                  bli_trmm_blk_var3f  },
		    { NULL,                 NULL,                  NULL                },
		}
	},
	// right
	{
		// lower
		{
		    // unblocked            optimized unblocked    blocked
		    { NULL,                 NULL,                  bli_trmm_blk_var1   },
		    { NULL,                 bli_trmm_rl_ker_var2,  bli_trmm_blk_var2f  },
		    { NULL,                 NULL,                  bli_trmm_blk_var3f  },
		    { NULL,                 NULL,                  NULL                },
		},
		// upper
		{
		    // unblocked            optimized unblocked    blocked
		    { NULL,                 NULL,                  bli_trmm_blk_var1   },
		    { NULL,                 bli_trmm_ru_ker_var2,  bli_trmm_blk_var2b  },
		    { NULL,                 NULL,                  bli_trmm_blk_var3b  },
		    { NULL,                 NULL,                  NULL                },
		}
	}
};

void bli_trmm_int( obj_t*  alpha,
=======
static FUNCPTR_T vars[2][5][3] =
{
	// lower
	{
	    // unblocked            optimized unblocked    blocked
	    { NULL,                 NULL,                  bli_trmm_l_blk_var1 },
	    { NULL,                 bli_trmm_l_ker_var2,   bli_trmm_l_blk_var2 },
	    { NULL,                 NULL,                  bli_trmm_l_blk_var3 },
	    { NULL,                 NULL,                  bli_trmm_l_blk_var4 },
	    { NULL,                 NULL,                  NULL                },
	},
	// upper
	{
	    // unblocked            optimized unblocked    blocked
	    { NULL,                 NULL,                  bli_trmm_u_blk_var1 },
	    { NULL,                 bli_trmm_u_ker_var2,   bli_trmm_u_blk_var2 },
	    { NULL,                 NULL,                  bli_trmm_u_blk_var3 },
	    { NULL,                 NULL,                  bli_trmm_u_blk_var4 },
	    { NULL,                 NULL,                  NULL                },
	}
};

void bli_trmm_int( side_t  side,
                   obj_t*  alpha,
>>>>>>> 0c1c78278bbd9c281bcbe933cc2f3bdb3bd74ef1
                   obj_t*  a,
                   obj_t*  b,
                   obj_t*  beta,
                   obj_t*  c,
                   trmm_t* cntl )
{
	obj_t     c_local;
<<<<<<< HEAD
	bool_t    side, uplo;
=======
	bool_t    uplo;
>>>>>>> 0c1c78278bbd9c281bcbe933cc2f3bdb3bd74ef1
	varnum_t  n;
	impl_t    i;
	FUNCPTR_T f;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
<<<<<<< HEAD
		bli_trmm_int_check( alpha, a, b, beta, c, cntl );

	// If C has a zero dimension, return early.
	if ( bli_obj_has_zero_dim( *c ) ) return;

	// If A or B has a zero dimension, scale C by beta and return early.
	if ( bli_obj_has_zero_dim( *a ) ||
	     bli_obj_has_zero_dim( *b ) )
	{
		bli_scalm( beta, c );
		return;
	}

=======
		bli_trmm_int_check( side, alpha, a, b, beta, c, cntl );

	// Return early if one of the matrix operands has a zero dimension.
	if ( bli_obj_has_zero_dim( *a ) ) return;
	if ( bli_obj_has_zero_dim( *b ) ) return;
	if ( bli_obj_has_zero_dim( *c ) ) return;

>>>>>>> 0c1c78278bbd9c281bcbe933cc2f3bdb3bd74ef1
	// Alias C in case we need to induce a transposition.
	bli_obj_alias_to( *c, c_local );

	// If we are about to call a leaf-level implementation, and matrix C
	// still needs a transposition, then we must induce one by swapping the
	// strides and dimensions. Note that this transposition would normally
	// be handled explicitly in the packing of C, but if C is not being
	// packed, this is our last chance to handle the transposition.
	if ( cntl_is_leaf( cntl ) && bli_obj_has_trans( *c ) )
	{
		bli_obj_induce_trans( c_local );
<<<<<<< HEAD
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, c_local );
	}

	// Set two bools: one based on the implied side parameter (the structure
	// of the root object) and one based on the uplo field of the triangular
	// matrix's root object (whether that is matrix A or matrix B).
	if ( bli_obj_root_is_triangular( *a ) )
	{
		side = 0;
		if ( bli_obj_root_is_lower( *a ) ) uplo = 0;
		else                               uplo = 1;
	}
	else // if ( bli_obj_root_is_triangular( *b ) )
	{
		side = 1;
		// Set a bool based on the uplo field of A's root object.
		if ( bli_obj_root_is_lower( *b ) ) uplo = 0;
		else                               uplo = 1;
	}
=======
		bli_obj_set_trans( BLIS_NO_TRANSPOSE, c_local );
	}

	// Set a bool based on the uplo field of A's root object.
	if ( bli_obj_root_is_lower( *a ) ) uplo = 0;
	else                               uplo = 1;
>>>>>>> 0c1c78278bbd9c281bcbe933cc2f3bdb3bd74ef1

	// Extract the variant number and implementation type.
	n = cntl_var_num( cntl );
	i = cntl_impl_type( cntl );

	// Index into the variant array to extract the correct function pointer.
<<<<<<< HEAD
	f = vars[side][uplo][n][i];
=======
	f = vars[uplo][n][i];
>>>>>>> 0c1c78278bbd9c281bcbe933cc2f3bdb3bd74ef1

	// Invoke the variant.
	f( alpha,
	   a,
	   b,
	   beta,
	   &c_local,
	   cntl );
}

