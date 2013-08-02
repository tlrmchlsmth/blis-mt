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
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef BLIS_KERNEL_H
#define BLIS_KERNEL_H


#define BLOMP_THREADS_L1  1
#define BLOMP_THREADS_L2  4
#define BLOMP_THREADS_L3  4
#define BLOMP_THREADS_L4  1

#if BLOMP_THREADS_L1 > 1
#define BLOMP_L1_PRAGMA _Pragma( "omp parallel for " )
#else
#define BLOMP_L1_PRAGMA 
#endif

#define BLOMP_L2_PRAGMA _Pragma( "omp parallel num_threads(BLOMP_THREADS_L2)" )
#define BLOMP_L2_PACK_PRAGMA _Pragma( "omp parallel num_threads(ONT)" )

#if BLOMP_THREADS_L3 > 1
#define BLOMP_L3_PRAGMA _Pragma( "omp parallel num_threads(BLOMP_THREADS_L3)" )
#else
#define BLOMP_L3_PRAGMA
#endif

#if BLOMP_THREADS_L4 > 1
#define BLOMP_L4_PRAGMA _Pragma( "omp parallel num_threads(BLOMP_THREADS_L4)" )
#else
#define BLOMP_L4_PRAGMA
#endif


// -- LEVEL-3 MICRO-KERNEL CONSTANTS -------------------------------------------

// -- Default cache blocksizes --

// Constraints:
//
// (1) MC must be a multiple of:
//     (a) MR (for zero-padding purposes) and
//     (b) NR.
// (2) NC must be a multiple of
//     (a) NR (for zero-padding purposes) and
//     (b) MR.
// (3) KC does not need to be multiple of anything, unless the micro-kernel
//     specifically requires it (and typically it does not).
// 
// NOTE: For BLIS libraries built on block-panel macro-kernels, constraint
// (2b) is relaxed. In this case, (1b) is needed for operation implementations
// involving matrices with diagonals (trmm, trsm). In these cases, we want the
// diagonal offset of any panel of packed matrix A to have a diagonal offset
// that is a multiple of MR. If, instead, the library were to be built on
// block-panel macro-kernels, matrix B would be the one with structure, not A,
// and thus it would be constraint (2b) that would be needed instead of (1b).
//

#define BLIS_DEFAULT_MC_S              128
#define BLIS_DEFAULT_KC_S              256
#define BLIS_DEFAULT_NC_S              8192

#define BLIS_DEFAULT_MC_D              96
#define BLIS_DEFAULT_KC_D              384
#define BLIS_DEFAULT_NC_D              8192

#define BLIS_DEFAULT_MC_C              128
#define BLIS_DEFAULT_KC_C              256
#define BLIS_DEFAULT_NC_C              8192

#define BLIS_DEFAULT_MC_Z              128
#define BLIS_DEFAULT_KC_Z              256
#define BLIS_DEFAULT_NC_Z              8192

// -- Default register blocksizes for inner kernel --

// NOTE: When using the reference configuration, these register blocksizes
// in the m and n dimensions should all be equal to the size expected by
// the reference micro-kernel(s).

#define BLIS_DEFAULT_MR_S              4
#define BLIS_DEFAULT_NR_S              4

#define BLIS_DEFAULT_MR_D              8
#define BLIS_DEFAULT_NR_D              4

#define BLIS_DEFAULT_MR_C              4
#define BLIS_DEFAULT_NR_C              4

#define BLIS_DEFAULT_MR_Z              4
#define BLIS_DEFAULT_NR_Z              4

// NOTE: If the micro-kernel, which is typically unrolled to a factor
// of f, handles leftover edge cases (ie: when k % f > 0) then these
// register blocksizes in the k dimension can be defined to 1.

#define BLIS_DEFAULT_KR_S              1
#define BLIS_DEFAULT_KR_D              1
#define BLIS_DEFAULT_KR_C              1
#define BLIS_DEFAULT_KR_Z              1

// -- Number of elements per vector register --

// NOTE: These constants are typically only used to determine the amount
// of duplication needed when configuring level-3 macro-kernels that
// copy and duplicate elements of B to a temporary duplication buffer
// (so that element-wise vector multiplication and addition instructions
// can be used).

#define BLIS_NUM_ELEM_PER_REG_S        4
#define BLIS_NUM_ELEM_PER_REG_D        4
#define BLIS_NUM_ELEM_PER_REG_C        2
#define BLIS_NUM_ELEM_PER_REG_Z        1

// -- Default switch for duplication of B --

// NOTE: If BLIS_DEFAULT_DUPLICATE_B is set to FALSE, then the
// NUM_DUPL definitions are not used.

//#define BLIS_DEFAULT_DUPLICATE_B       TRUE
#define BLIS_DEFAULT_DUPLICATE_B       FALSE
#define BLIS_DEFAULT_NUM_DUPL_S        BLIS_NUM_ELEM_PER_REG_S
#define BLIS_DEFAULT_NUM_DUPL_D        BLIS_NUM_ELEM_PER_REG_D
#define BLIS_DEFAULT_NUM_DUPL_C        BLIS_NUM_ELEM_PER_REG_C
#define BLIS_DEFAULT_NUM_DUPL_Z        BLIS_NUM_ELEM_PER_REG_Z

// -- Default incremental packing blocksizes (n dimension) --

// NOTE: These incremental packing blocksizes (for the n dimension) are only
// used by certain blocked variants. But when the *are* used, they MUST be
// be an integer multiple of NR!

#define BLIS_DEFAULT_NI_FAC            16
#define BLIS_DEFAULT_NI_S              (BLIS_DEFAULT_NI_FAC * BLIS_DEFAULT_NR_S)
#define BLIS_DEFAULT_NI_D              (BLIS_DEFAULT_NI_FAC * BLIS_DEFAULT_NR_D)
#define BLIS_DEFAULT_NI_C              (BLIS_DEFAULT_NI_FAC * BLIS_DEFAULT_NR_C)
#define BLIS_DEFAULT_NI_Z              (BLIS_DEFAULT_NI_FAC * BLIS_DEFAULT_NR_Z)



// -- LEVEL-1F KERNEL CONSTANTS ------------------------------------------------

// -- Default fusing factors for level-1f operations --

// NOTE: Default fusing factors are not used by the reference implementations
// of level-1f operations. They are here only for use when these operations
// are optimized.

#define BLIS_DEFAULT_FUSING_FACTOR_S   8
#define BLIS_DEFAULT_FUSING_FACTOR_D   4
#define BLIS_DEFAULT_FUSING_FACTOR_C   4
#define BLIS_DEFAULT_FUSING_FACTOR_Z   2



// -- LEVEL-1V KERNEL CONSTANTS ------------------------------------------------

// -- Default register blocksizes for vectors --

// NOTE: Register blocksizes for vectors are used when packing
// non-contiguous vectors. Similar to that of KR, they can
// typically be set to 1.

#define BLIS_DEFAULT_VR_S              1
#define BLIS_DEFAULT_VR_D              1
#define BLIS_DEFAULT_VR_C              1
#define BLIS_DEFAULT_VR_Z              1



// -- LEVEL-3 KERNEL DEFINITIONS -----------------------------------------------

// -- dupl --

#define DUPL_KERNEL          dupl_unb_var1

// -- gemm --
#include "bl2_gemm_opt_8x4_ref_u4_nodupl_avx1.h"
#define GEMM_UKERNEL         gemm_opt_8x4_ref_u4_nodupl_avx1

// -- trsm-related --

#define GEMMTRSM_L_UKERNEL   gemmtrsm_l_ref_4x4
#define GEMMTRSM_U_UKERNEL   gemmtrsm_u_ref_4x4

#define TRSM_L_UKERNEL       trsm_l_ref_4x4
#define TRSM_U_UKERNEL       trsm_u_ref_4x4



// -- LEVEL-1M KERNEL DEFINITIONS ----------------------------------------------

// -- packm --

#define PACKM_2XK_KERNEL     packm_ref_2xk
#define PACKM_4XK_KERNEL     packm_ref_4xk
#define PACKM_6XK_KERNEL     packm_ref_6xk
#define PACKM_8XK_KERNEL     packm_ref_8xk
#define PACKM_10XK_KERNEL    packm_ref_10xk
#define PACKM_12XK_KERNEL    packm_ref_12xk
#define PACKM_14XK_KERNEL    packm_ref_14xk
#define PACKM_16XK_KERNEL    packm_ref_16xk

// -- unpackm --

#define UNPACKM_2XK_KERNEL   unpackm_ref_2xk
#define UNPACKM_4XK_KERNEL   unpackm_ref_4xk
#define UNPACKM_6XK_KERNEL   unpackm_ref_6xk
#define UNPACKM_8XK_KERNEL   unpackm_ref_8xk
#define UNPACKM_10XK_KERNEL  unpackm_ref_10xk
#define UNPACKM_12XK_KERNEL  unpackm_ref_12xk
#define UNPACKM_14XK_KERNEL  unpackm_ref_14xk
#define UNPACKM_16XK_KERNEL  unpackm_ref_16xk



// -- LEVEL-1F KERNEL DEFINITIONS ----------------------------------------------

// -- axpy2v --

#define AXPY2V_KERNEL        axpy2v_unb_var1

// -- dotaxpyv --

#define DOTAXPYV_KERNEL      dotaxpyv_unb_var1

// -- axpyf --

#define AXPYF_KERNEL         axpyf_unb_var1

// -- dotxf --

#define DOTXF_KERNEL         dotxf_unb_var1

// -- dotxaxpyf --

#define DOTXAXPYF_KERNEL     dotxaxpyf_unb_var1



// -- LEVEL-1V KERNEL DEFINITIONS ----------------------------------------------

// -- axpyv --

#define AXPYV_KERNEL         axpyv_unb_var1

// -- copynzv --

#define COPYNZV_KERNEL       copynzv_unb_var1

// -- copyv --

#define COPYV_KERNEL         copyv_unb_var1

// -- dotv --

#define DOTV_KERNEL          dotv_unb_var1

// -- dotxv --

#define DOTXV_KERNEL         dotxv_unb_var1

// -- invertv --

#define INVERTV_KERNEL       invertv_unb_var1

// -- scal2v --

#define SCAL2V_KERNEL        scal2v_unb_var1

// -- scalv --

#define SCALV_KERNEL         scalv_unb_var1

// -- setv --

#define SETV_KERNEL          setv_unb_var1



#endif

