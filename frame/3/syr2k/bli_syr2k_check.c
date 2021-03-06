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

void bli_syr2k_basic_check( obj_t*   alpha,
                            obj_t*   a,
                            obj_t*   bt,
                            obj_t*   b,
                            obj_t*   at,
                            obj_t*   beta,
                            obj_t*   c )
{
	// The basic properties of syr2k are identical to that of her2k.
	bli_her2k_basic_check( alpha, a, bt, alpha, b, at, beta, c );
}

void bli_syr2k_check( obj_t*   alpha,
                      obj_t*   a,
                      obj_t*   b,
                      obj_t*   beta,
                      obj_t*   c )
{
	err_t e_val;
	obj_t at, bt;

	// Alias A and B to A^T and B^T so we can perform dimension checks.
	bli_obj_alias_with_trans( BLIS_TRANSPOSE, *a, at );
	bli_obj_alias_with_trans( BLIS_TRANSPOSE, *b, bt );

	// Check basic properties of the operation.

	bli_syr2k_basic_check( alpha, a, &bt, b, &at, beta, c );

	// Check matrix squareness.

	e_val = bli_check_square_object( c );
	bli_check_error_code( e_val );

	// Check matrix structure.

	e_val = bli_check_symmetric_object( c );
	bli_check_error_code( e_val );
}

