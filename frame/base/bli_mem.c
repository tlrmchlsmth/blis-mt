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


// Define the size of pool blocks. These may be adjusted so that they can
// handle inflated blocksizes at edge cases.
#define BLIS_POOL_MC_D     ( ( BLIS_MAXIMUM_MC_D * BLIS_PACKDIM_MR_D ) / BLIS_DEFAULT_MR_D )
#define BLIS_POOL_KC_D     ( ( BLIS_MAXIMUM_KC_D * BLIS_PACKDIM_KR_D ) / BLIS_DEFAULT_KR_D )
#define BLIS_POOL_NC_D     ( ( BLIS_MAXIMUM_NC_D * BLIS_PACKDIM_NR_D ) / BLIS_DEFAULT_NR_D )

// Define each pool's block size.
// NOTE: Here we assume the "worst" case of the register blocking
// being unit and every row of A and column of B needing maximum
// padding to conform to the system alignment.
#define BLIS_MK_BLOCK_SIZE ( BLIS_POOL_MC_D * \
                             ( BLIS_POOL_KC_D + \
                               ( BLIS_CONTIG_STRIDE_ALIGN_SIZE / \
                                 sizeof( double ) \
                               ) \
                             ) * \
                             sizeof( double ) \
                           )
#define BLIS_KN_BLOCK_SIZE ( ( BLIS_POOL_KC_D + \
                               ( BLIS_CONTIG_STRIDE_ALIGN_SIZE / \
                                 sizeof( double ) \
                               ) \
                             ) * \
                             BLIS_POOL_NC_D * \
                             sizeof( double ) \
                           )
#define BLIS_MN_BLOCK_SIZE ( BLIS_POOL_MC_D * \
                             BLIS_POOL_NC_D * \
                             sizeof( double ) \
                           )

// Define each pool's total size.
#define BLIS_MK_POOL_SIZE  ( \
                             BLIS_NUM_MC_X_KC_BLOCKS * \
                             ( BLIS_MK_BLOCK_SIZE + \
                               BLIS_CONTIG_ADDR_ALIGN_SIZE \
                             ) + \
                             BLIS_MAX_PRELOAD_BYTE_OFFSET \
                           )

#define BLIS_KN_POOL_SIZE  ( \
                             BLIS_NUM_KC_X_NC_BLOCKS * \
                             ( BLIS_KN_BLOCK_SIZE + \
                               BLIS_CONTIG_ADDR_ALIGN_SIZE \
                             ) + \
                             BLIS_MAX_PRELOAD_BYTE_OFFSET \
                           )

#define BLIS_MN_POOL_SIZE  ( \
                             BLIS_NUM_MC_X_NC_BLOCKS * \
                             ( BLIS_MN_BLOCK_SIZE + \
                               BLIS_CONTIG_ADDR_ALIGN_SIZE \
                             ) + \
                             BLIS_MAX_PRELOAD_BYTE_OFFSET \
                           )

// Declare one memory pool structure for each block size/shape we want to
// be able to allocate.

static pool_t pools[3];


// Physically contiguous memory for each pool.

static void*  pool_mk_blk_ptrs[ BLIS_NUM_MC_X_KC_BLOCKS ];
static char   pool_mk_mem[ BLIS_MK_POOL_SIZE ];

static void*  pool_kn_blk_ptrs[ BLIS_NUM_KC_X_NC_BLOCKS ];
static char   pool_kn_mem[ BLIS_KN_POOL_SIZE ];

static void*  pool_mn_blk_ptrs[ BLIS_NUM_MC_X_NC_BLOCKS ];
static char   pool_mn_mem[ BLIS_MN_POOL_SIZE ];





void bli_mem_acquire_m( siz_t     req_size,
                        packbuf_t buf_type,
                        mem_t*    mem )
{
	siz_t   block_size;
	dim_t   pool_index;
	pool_t* pool;
	void**  block_ptrs;
	void*   block;
	int     i;


	if ( buf_type == BLIS_BUFFER_FOR_GEN_USE )
	{
		// For general-use buffer requests, such as those used by level-2
		// operations, using bli_malloc() is sufficient, since using
		// physically contiguous memory is not as important there.
		block = bli_malloc( req_size );

		// Initialize the mem_t object with:
		// - the address of the memory block,
		// - the buffer type (a packbuf_t value), and
		// - the size of the requested region.
		// NOTE: We do not initialize the pool field since this block did not
		// come from a contiguous memory pool.
		bli_mem_set_buffer( block, mem );
		bli_mem_set_buf_type( buf_type, mem );
		bli_mem_set_size( req_size, mem );
	}
	else
	{
		// This branch handles cases where the memory block needs to come
		// from one of the contiguous memory pools.

		// Map the requested packed buffer type to a zero-based index, which
		// we then use to select the corresponding memory pool.
		pool_index = bli_packbuf_index( buf_type );
		pool       = &pools[ pool_index ];

		// Perform error checking, if enabled.
		if ( bli_error_checking_is_enabled() )
		{
			err_t e_val;

			// Make sure that the requested matrix size fits inside of a block
			// of the corresponding pool.
			e_val = bli_check_requested_block_size_for_pool( req_size, pool );
			bli_check_error_code( e_val );

			// Make sure that the pool contains at least one block to check out
			// to the thread.
			e_val = bli_check_if_exhausted_pool( pool );
			bli_check_error_code( e_val );
		}

		// Access the block pointer array from the memory pool data structure.
		block_ptrs = bli_pool_block_ptrs( pool );


		// BEGIN CRITICAL SECTION
        _Pragma( "omp critical(mem)" ) {

		// Query the index of the contiguous memory block that resides at the
		// "top" of the pool.
		i = bli_pool_top_index( pool );
	
		// Extract the address of the top block from the block pointer array.
		block = block_ptrs[i];

		// Clear the entry from the block pointer array. (This is actually not
		// necessary.)
		//block_ptrs[i] = NULL; 

		// Decrement the top of the memory pool.
		bli_pool_dec_top_index( pool );


		// END CRITICAL SECTION
        }

		// Query the size of the blocks in the pool so we can store it in the
		// mem_t object.
		block_size = bli_pool_block_size( pool );

		// Initialize the mem_t object with:
		// - the address of the memory block,
		// - the buffer type (a packbuf_t value),
		// - the address of the memory pool to which it belongs, and
		// - the size of the contiguous memory block (NOT the size of the
		//   requested region).
		bli_mem_set_buffer( block, mem );
		bli_mem_set_buf_type( buf_type, mem );
		bli_mem_set_pool( pool, mem );
		bli_mem_set_size( block_size, mem );
	}
}


void bli_mem_release( mem_t* mem )
{
	packbuf_t buf_type;
	pool_t*   pool;
	void**    block_ptrs;
	void*     block;
	int       i;

	// Extract the address of the memory block we are trying to
	// release.
	block = bli_mem_buffer( mem );

	// Extract the buffer type so we know what kind of memory was allocated.
	buf_type = bli_mem_buf_type( mem );

	if ( buf_type == BLIS_BUFFER_FOR_GEN_USE )
	{
		// For general-use buffers, we allocate with bli_malloc(), and so
		// here we need to call bli_free().
		bli_free( block );
	}
	else
	{
		// This branch handles cases where the memory block came from one
		// of the contiguous memory pools.

		// Extract the pool from which the block was allocated.
		pool = bli_mem_pool( mem );

		// Extract the block pointer array associated with the pool.
		block_ptrs = bli_pool_block_ptrs( pool );


		// BEGIN CRITICAL SECTION
        _Pragma( "omp critical(mem)" ) {


		// Increment the top of the memory pool.
		bli_pool_inc_top_index( pool );

		// Query the newly incremented top index.
		i = bli_pool_top_index( pool );

		// Place the address of the block back onto the top of the memory pool.
		block_ptrs[i] = block;


		// END CRITICAL SECTION
        }
	}


	// Clear the mem_t object so that it appears unallocated. We clear:
	// - the buffer field,
	// - the pool field, and
	// - the size field.
	// NOTE: We do not clear the buf_type field since there is no
	// "uninitialized" value for packbuf_t.
	bli_mem_set_buffer( NULL, mem );
	bli_mem_set_pool( NULL, mem );
	bli_mem_set_size( 0, mem );
}


void bli_mem_acquire_v( siz_t  req_size,
                        mem_t* mem )
{
	bli_mem_acquire_m( req_size,
	                   BLIS_BUFFER_FOR_GEN_USE,
	                   mem );
}



void bli_mem_init()
{
	dim_t index_a;
	dim_t index_b;
	dim_t index_c;

	// Map each of the packbuf_t values to an index starting at zero.
	index_a = bli_packbuf_index( BLIS_BUFFER_FOR_A_BLOCK );
	index_b = bli_packbuf_index( BLIS_BUFFER_FOR_B_PANEL );
	index_c = bli_packbuf_index( BLIS_BUFFER_FOR_C_PANEL );

	// Initialize contiguous memory pool for MC x KC blocks.
	bli_mem_init_pool( pool_mk_mem,
	                   BLIS_MK_BLOCK_SIZE,
	                   BLIS_NUM_MC_X_KC_BLOCKS,
	                   pool_mk_blk_ptrs,
	                   &pools[ index_a ] );

	// Initialize contiguous memory pool for KC x NC blocks.
	bli_mem_init_pool( pool_kn_mem,
	                   BLIS_KN_BLOCK_SIZE,
	                   BLIS_NUM_KC_X_NC_BLOCKS,
	                   pool_kn_blk_ptrs,
	                   &pools[ index_b ] );

	// Initialize contiguous memory pool for MC x NC blocks.
	bli_mem_init_pool( pool_mn_mem,
	                   BLIS_MN_BLOCK_SIZE,
	                   BLIS_NUM_MC_X_NC_BLOCKS,
	                   pool_mn_blk_ptrs,
	                   &pools[ index_c ] );
}


void bli_mem_init_pool( char*   pool_mem,
                        siz_t   block_size,
                        dim_t   num_blocks,
                        void**  block_ptrs,
                        pool_t* pool )
{
	const siz_t align_size = BLIS_CONTIG_ADDR_ALIGN_SIZE;
	dim_t       i;

	// If the pool starting address is not already aligned, advance it
	// accordingly.
	if ( bli_is_unaligned_to( pool_mem, align_size ) )
	{
		// Notice that this works even if the alignment is not a power of two.
		pool_mem += ( align_size - 
		              ( ( siz_t )pool_mem % align_size ) );
	}

	// Step through the memory pool, beginning with the aligned address
	// determined above, assigning pointers to the beginning of each block_size
	// bytes to the ith element of the block_ptrs array.
	for ( i = 0; i < num_blocks; ++i )
	{
		// Save the address of pool, which is guaranteed to be aligned.
		block_ptrs[i] = pool_mem;

		// Advance pool by one block.
		pool_mem += block_size;

		// Advance pool a bit further if needed in order to get to the
		// beginning of an alignment boundary.
		if ( bli_is_unaligned_to( pool_mem, align_size ) )
		{
			pool_mem += ( align_size -
			              ( ( siz_t )pool_mem % align_size ) );
		}
	}

	// Now that we have initialized the array of pointers to the individual
	// blocks in the pool, we initialize a pool_t data structure so that we
	// can easily manage this pool.
	bli_pool_init( num_blocks,
	               block_size,
	               block_ptrs,
	               pool );
}



void bli_mem_finalize()
{
	// Nothing to do.
}

