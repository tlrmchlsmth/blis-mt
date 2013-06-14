/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

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
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
   OF TEXAS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
   OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#undef restrict

void bli_dgemm_16x4( dim_t k, double* alpha, double* a, double* b, double* beta, double* c, inc_t rs_c, inc_t cs_c, double* a_next, double* b_next );
void bli_sgemm_16x16(
                        dim_t     k,
                        float*    alpha,
                        float*    a,
                        float*    b,
                        float*    beta,
                        float*    c, inc_t rs_c, inc_t cs_c,
                        float* a_next, float* b_next
                      )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}


/*
 * Here is dgemm kernel for QPX. 
 * Instruction mix was divined by a statement in an email from John Gunnels when asked about the peak performance with a single thread:
 * "Achievable peak can either be:
 * 1) 12.8 GF 8 FMAs cycle * 1.6 GHz
 * 2) 8.53 GF Takes into account the instruction mix in DGEMM and the fact that you can only do an FMA or a load/store in a single cycle with just one thread
 * 3) 7.58 GF (2) + the fact that we can only issue 8 instructions in 9 cycles with one thread"
 *
 * Which I have taken to mean: 8.53 GFLOPS implies on average 5.33 flops/cycle. 
 * I know the kernel John uses is 8x8, so 16 flops per loop iteration. 
 * Thus there must be 24 total instructions per iteration because 16/24 = 5.33.
 *
 * We have 4 loads per iteration, so we have 4 more instructions to play with. 2 are permutations of B,
 * and we use the xmadd and xxmadd instructions to effectively gain free permutations of B.
*/
void bli_dgemm_8x8(
                        dim_t     k,
                        restrict double*   alpha,
                        restrict double*   a,
                        restrict double*   b,
                        restrict double*   beta,
                        restrict double*   c, inc_t rs_c, inc_t cs_c,
                        dim_t ap_offset, dim_t bp_offset
                      )

{
    //Registers for storing C.
    //4 4x4 subblocks of C, c00, c01, c10, c11
    //4 registers per subblock: a, b, c, d
    //There is an excel file that details which register ends up storing what
    vector4double c00a = vec_splats( 0.0 );
    vector4double c00b = vec_splats( 0.0 );
    vector4double c00c = vec_splats( 0.0 );
    vector4double c00d = vec_splats( 0.0 );

    vector4double c01a = vec_splats( 0.0 );
    vector4double c01b = vec_splats( 0.0 );
    vector4double c01c = vec_splats( 0.0 );
    vector4double c01d = vec_splats( 0.0 );

    vector4double c10a = vec_splats( 0.0 );
    vector4double c10b = vec_splats( 0.0 );
    vector4double c10c = vec_splats( 0.0 );
    vector4double c10d = vec_splats( 0.0 );

    vector4double c11a = vec_splats( 0.0 );
    vector4double c11b = vec_splats( 0.0 );
    vector4double c11c = vec_splats( 0.0 );
    vector4double c11d = vec_splats( 0.0 );

    vector4double a0, a1;
    vector4double b0, b1;
    vector4double b0p, b1p;

    vector4double pattern = vec_gpci( 02301 );
    //a_prefetch += PREFETCH_OFFSET;
    //b_prefetch += PREFETCH_OFFSET;
   
    for( dim_t i = 0; i < k; i++ )
    {
        a0 = vec_lda( 0 * sizeof(double), a );
        b0 = vec_lda( 0 * sizeof(double), b );
        b1 = vec_lda( 4 * sizeof(double), b );
        a1 = vec_lda( 4 * sizeof(double), a );

        __dcbt( a + ap_offset);
        __dcbt( b + bp_offset);
        
        c00a    = vec_xmadd( b0, a0, c00a );
        c00b    = vec_xxmadd( a0, b0, c00b );
        b0p     = vec_perm( b0, b0, pattern ); 
        c00c    = vec_xmadd( b0p, a0, c00c );
        c00d    = vec_xxmadd( a0, b0p, c00d );

        
        c01a    = vec_xmadd( b1, a0, c01a );
        c01b    = vec_xxmadd( a0, b1, c01b );
        b1p     = vec_perm( b1, b1, pattern ); 
        c01c    = vec_xmadd( b1p, a0, c01c );
        c01d    = vec_xxmadd( a0, b1p, c01d );


        c10a    = vec_xmadd( b0, a1, c10a );
        c10b    = vec_xxmadd( a1, b0, c10b );
        c10c    = vec_xmadd( b0p, a1, c10c );
        c10d    = vec_xxmadd( a1, b0p, c10d );

        c11a    = vec_xmadd( b1, a1, c11a );
        c11b    = vec_xxmadd( a1, b1, c11b );
        c11c    = vec_xmadd( b1p, a1, c11c );
        c11d    = vec_xxmadd( a1, b1p, c11d );

        a += 8*2;
        b += 8*2;
        //a_prefetch += 16;
        //b_prefetch += 16;
    }
    
    // Create patterns for permuting C
    vector4double patternCA = vec_gpci( 00167 );
    vector4double patternCC = vec_gpci( 04523 );
    vector4double patternCB = vec_gpci( 01076 );
    vector4double patternCD = vec_gpci( 05432 );

    vector4double AB;
    vector4double C = vec_splats( 0.0 );
    vector4double betav = vec_splats( *beta );
    vector4double alphav = vec_splats( *alpha );
    double ct;
    
#define UPDATE( REG, ADDR, OFFSET )     \
{                                       \
    ct = *(ADDR + OFFSET);              \
    C = vec_insert( ct, C, 0 );         \
    ct = *(ADDR + OFFSET + 1);          \
    C = vec_insert( ct, C, 1 );         \
    ct = *(ADDR + OFFSET + 2 );         \
    C = vec_insert( ct, C, 2 );         \
    ct = *(ADDR + OFFSET + 3 );         \
    C = vec_insert( ct, C, 3 );         \
                                        \
    AB = vec_mul( REG, alphav );        \
    AB = vec_madd( C, betav, AB);       \
                                        \
    ct = vec_extract( AB, 0 );          \
    *(ADDR + OFFSET) = ct;              \
    ct = vec_extract( AB, 1 );          \
    *(ADDR + OFFSET + 1) = ct;          \
    ct = vec_extract( AB, 2 );          \
    *(ADDR + OFFSET + 2) = ct;          \
    ct = vec_extract( AB, 3 );          \
    *(ADDR + OFFSET + 3) = ct;          \
}  
    //Update c00 and c10 sub-blocks
    AB = vec_perm( c00a, c00c, patternCA );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c10a, c10c, patternCA );
    UPDATE( AB, c, 4 );

    c = c + cs_c;
    AB = vec_perm( c00b, c00d, patternCB );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c10b, c10d, patternCB );
    UPDATE( AB, c, 4 );

    c = c + cs_c;
    AB = vec_perm( c00a, c00c, patternCC );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c10a, c10c, patternCC );
    UPDATE( AB, c, 4 );

    c = c + cs_c;
    AB = vec_perm( c00b, c00d, patternCD );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c10b, c10d, patternCD );
    UPDATE( AB, c, 4 );

    //Update c01 and c11 sub-blocks
    c = c + cs_c;
    AB = vec_perm( c01a, c01c, patternCA );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c11a, c11c, patternCA );
    UPDATE( AB, c, 4 );

    c = c + cs_c;
    AB = vec_perm( c01b, c01d, patternCB );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c11b, c11d, patternCB );
    UPDATE( AB, c, 4 );

    c = c + cs_c;
    AB = vec_perm( c01a, c01c, patternCC );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c11a, c11c, patternCC );
    UPDATE( AB, c, 4 );

    c = c + cs_c;
    AB = vec_perm( c01b, c01d, patternCD );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c11b, c11d, patternCD );
    UPDATE( AB, c, 4 );
}

#define PREFETCH_OFFSET 320
#define USE_8X8
void bli_dgemm_16x16_mt(
                        dim_t     k,
                        restrict double*   alpha,
                        restrict double*   a,
                        restrict double*   b,
                        restrict double*   beta,
                        restrict double*   c, inc_t rs_c, inc_t cs_c,
                        restrict double* a_next, restrict double* b_next,
                        int tid
                      )
{
#ifdef USE_8X8
    int m_tid = tid >> 1;
    int n_tid = tid & 1;
    double * a_addr = a + 8 * m_tid;
    double * b_addr = b + 8 * n_tid;
    bli_dgemm_8x8( k, alpha, 
        a_addr, 
        b_addr, beta, 
        c + 8 * m_tid * rs_c + 8 * n_tid * cs_c, 
        rs_c, cs_c, //NULL, NULL);
        8 + 4 * n_tid + PREFETCH_OFFSET, 
        8 + 4 * m_tid + PREFETCH_OFFSET); 
#else
    bli_dgemm_16x4( k, alpha, 
        a,
        b + 4 * tid, beta, 
        c + 4 * tid * cs_c, 
        rs_c, cs_c, NULL, NULL );
#endif
}

void bli_dgemm_16x16(
                        dim_t     k,
                        restrict double*   alpha,
                        restrict double*   a,
                        restrict double*   b,
                        restrict double*   beta,
                        restrict double*   c, inc_t rs_c, inc_t cs_c,
                        restrict double* a_next, restrict double* b_next
                      )
{
    bli_dgemm_16x16_mt(k, alpha, a, b, beta, c, rs_c, cs_c, a_next, b_next, 0);
    bli_dgemm_16x16_mt(k, alpha, a, b, beta, c, rs_c, cs_c, a_next, b_next, 1);
    bli_dgemm_16x16_mt(k, alpha, a, b, beta, c, rs_c, cs_c, a_next, b_next, 2);
    bli_dgemm_16x16_mt(k, alpha, a, b, beta, c, rs_c, cs_c, a_next, b_next, 3);
}
void bli_dgemm_16x4(
                        dim_t     k,
                        double*   alpha,
                        double*   a,
                        double*   b,
                        double*   beta,
                        double*   c, inc_t rs_c, inc_t cs_c,
                        double* a_next, double* b_next
                      )

{
    //Registers for storing C.
    //4 4x4 subblocks of C, c00, c01, c10, c11
    //4 registers per subblock: a, b, c, d
    //There is an excel file that details which register ends up storing what
    vector4double c00a = vec_splats( 0.0 );
    vector4double c00b = vec_splats( 0.0 );
    vector4double c00c = vec_splats( 0.0 );
    vector4double c00d = vec_splats( 0.0 );

    vector4double c10a = vec_splats( 0.0 );
    vector4double c10b = vec_splats( 0.0 );
    vector4double c10c = vec_splats( 0.0 );
    vector4double c10d = vec_splats( 0.0 );

    vector4double c20a = vec_splats( 0.0 );
    vector4double c20b = vec_splats( 0.0 );
    vector4double c20c = vec_splats( 0.0 );
    vector4double c20d = vec_splats( 0.0 );

    vector4double c30a = vec_splats( 0.0 );
    vector4double c30b = vec_splats( 0.0 );
    vector4double c30c = vec_splats( 0.0 );
    vector4double c30d = vec_splats( 0.0 );

    vector4double a0, a1, a2, a3;
    vector4double b0;
    vector4double b0p;

    vector4double pattern = vec_gpci( 02301 );
    
    for( dim_t i = 0; i < k; i++ )
    {
        a0 = vec_lda(  0 * sizeof(double), a );
        a1 = vec_lda(  4 * sizeof(double), a );
        a2 = vec_lda(  8 * sizeof(double), a );
        a3 = vec_lda( 12 * sizeof(double), a );

        b0 = vec_lda( 0 * sizeof(double), b );
        
        c00a    = vec_xmadd( b0, a0, c00a );
        c00b    = vec_xxmadd( a0, b0, c00b );
        b0p     = vec_perm( b0, b0, pattern ); 
        c00c    = vec_xmadd( b0p, a0, c00c );
        c00d    = vec_xxmadd( a0, b0p, c00d );
        
        c10a    = vec_xmadd( b0, a1, c10a );
        c10b    = vec_xxmadd( a1, b0, c10b );
        c10c    = vec_xmadd( b0p, a1, c10c );
        c10d    = vec_xxmadd( a1, b0p, c10d );
        
        c20a    = vec_xmadd( b0, a2, c20a );
        c20b    = vec_xxmadd( a2, b0, c20b );
        c20c    = vec_xmadd( b0p, a2, c20c );
        c20d    = vec_xxmadd( a2, b0p, c20d );
        
        c30a    = vec_xmadd( b0, a3, c30a );
        c30b    = vec_xxmadd( a3, b0, c30b );
        c30c    = vec_xmadd( b0p, a3, c30c );
        c30d    = vec_xxmadd( a3, b0p, c30d );

        a += 16;
        b += 16;
    }
    
    // Create patterns for permuting C
    vector4double patternCA = vec_gpci( 00167 );
    vector4double patternCC = vec_gpci( 04523 );
    vector4double patternCB = vec_gpci( 01076 );
    vector4double patternCD = vec_gpci( 05432 );

    vector4double AB;
    vector4double C = vec_splats( 0.0 );
    vector4double betav = vec_splats( *beta );
    vector4double alphav = vec_splats( *alpha );
    double ct;
   
    //Update c00 and c10 sub-blocks
    AB = vec_perm( c00a, c00c, patternCA );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c10a, c10c, patternCA );
    UPDATE( AB, c, 4 );
    AB = vec_perm( c20a, c20c, patternCA );
    UPDATE( AB, c, 8 );
    AB = vec_perm( c30a, c30c, patternCA );
    UPDATE( AB, c, 12 );

    c = c + cs_c;

    AB = vec_perm( c00b, c00d, patternCB );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c10b, c10d, patternCB );
    UPDATE( AB, c, 4 );
    AB = vec_perm( c20b, c20d, patternCB );
    UPDATE( AB, c, 8 );
    AB = vec_perm( c30b, c30d, patternCB );
    UPDATE( AB, c, 12 );

    c = c + cs_c;

    AB = vec_perm( c00a, c00c, patternCC );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c10a, c10c, patternCC );
    UPDATE( AB, c, 4 );
    AB = vec_perm( c20a, c20c, patternCC );
    UPDATE( AB, c, 8 );
    AB = vec_perm( c30a, c30c, patternCC );
    UPDATE( AB, c, 12 );

    c = c + cs_c;

    AB = vec_perm( c00b, c00d, patternCD );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c10b, c10d, patternCD );
    UPDATE( AB, c, 4 );
    AB = vec_perm( c20b, c20d, patternCD );
    UPDATE( AB, c, 8 );
    AB = vec_perm( c30b, c30d, patternCD );
    UPDATE( AB, c, 12 );

}
void bli_cgemm_16x16(
                        dim_t     k,
                        scomplex* alpha,
                        scomplex* a,
                        scomplex* b,
                        scomplex* beta,
                        scomplex* c, inc_t rs_c, inc_t cs_c,
                        scomplex* a_next, scomplex* b_next
                      )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_zgemm_16x16(
                        dim_t     k,
                        dcomplex* alpha,
                        dcomplex* a,
                        dcomplex* b,
                        dcomplex* beta,
                        dcomplex* c, inc_t rs_c, inc_t cs_c,
                        dcomplex* a_next, dcomplex* b_next
                      )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}


void bli_sgemm_16x16_mt(
                        dim_t     k,
                        float*    alpha,
                        float*    a,
                        float*    b,
                        float*    beta,
                        float*    c, inc_t rs_c, inc_t cs_c,
                        float* a_next, float* b_next,
                        int t_id
                      )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_cgemm_16x16_mt(
                        dim_t     k,
                        scomplex* alpha,
                        scomplex* a,
                        scomplex* b,
                        scomplex* beta,
                        scomplex* c, inc_t rs_c, inc_t cs_c,
                        scomplex* a_next, scomplex* b_next,
                        int t_id
                      )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_zgemm_16x16_mt(
                        dim_t     k,
                        dcomplex* alpha,
                        dcomplex* a,
                        dcomplex* b,
                        dcomplex* beta,
                        dcomplex* c, inc_t rs_c, inc_t cs_c,
                        dcomplex* a_next, dcomplex* b_next,
                        int t_id
                      )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}
