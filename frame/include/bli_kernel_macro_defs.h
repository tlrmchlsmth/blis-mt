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

#ifndef BLIS_KERNEL_MACRO_DEFS_H
#define BLIS_KERNEL_MACRO_DEFS_H

#define SIZEOF_S  4
#define SIZEOF_D  8
#define SIZEOF_C  8
#define SIZEOF_Z  16


// -- Kernel macro checks ------------------------------------------------------

// Verify that cache blocksizes are whole multiples of register blocksizes.
// Specifically, verify that:
//   - MC is a whole multiple of MR.
//   - NC is a whole multiple of NR.
//   - KC is a whole multiple of KR.
// These constraints are enforced because it makes it easier to handle diagonals
// in the macro-kernel implementations. 
#if ( \
      ( BLIS_DEFAULT_MC_S % BLIS_DEFAULT_MR_S != 0 ) || \
      ( BLIS_DEFAULT_MC_D % BLIS_DEFAULT_MR_D != 0 ) || \
      ( BLIS_DEFAULT_MC_C % BLIS_DEFAULT_MR_C != 0 ) || \
      ( BLIS_DEFAULT_MC_Z % BLIS_DEFAULT_MR_Z != 0 )    \
    )
  #error MC must be multiple of MR for all datatypes.
#endif

#if ( \
      ( BLIS_DEFAULT_NC_S % BLIS_DEFAULT_NR_S != 0 ) || \
      ( BLIS_DEFAULT_NC_D % BLIS_DEFAULT_NR_D != 0 ) || \
      ( BLIS_DEFAULT_NC_C % BLIS_DEFAULT_NR_C != 0 ) || \
      ( BLIS_DEFAULT_NC_Z % BLIS_DEFAULT_NR_Z != 0 )    \
    )
  #error NC must be multiple of NR for all datatypes.
#endif

#if ( \
      ( BLIS_DEFAULT_KC_S % BLIS_DEFAULT_KR_S != 0 ) || \
      ( BLIS_DEFAULT_KC_D % BLIS_DEFAULT_KR_D != 0 ) || \
      ( BLIS_DEFAULT_KC_C % BLIS_DEFAULT_KR_C != 0 ) || \
      ( BLIS_DEFAULT_KC_Z % BLIS_DEFAULT_KR_Z != 0 )    \
    )
  #error KC must be multiple of KR for all datatypes.
#endif

/*
// Verify that cache blocksizes indicate consistent storage.
// Specifically, verify that:
//   - MC_D * KC_D >= MC_? * KC_?.
//   - KC_D * NC_D >= KC_? * NC_?.
//   - MC_D * NC_D >= MC_? * NC_?.
// These constraints are enforced because static memory is allocated for the
// contiguous memory allocator using the double-precision real values of MC,
// NC, and KC.
#if ( \
      ( ( BLIS_DEFAULT_MC_D * BLIS_DEFAULT_KC_D * SIZEOF_D ) < \
        ( BLIS_DEFAULT_MC_S * BLIS_DEFAULT_KC_S * SIZEOF_S ) ) || \
      ( ( BLIS_DEFAULT_MC_D * BLIS_DEFAULT_KC_D * SIZEOF_D ) < \
        ( BLIS_DEFAULT_MC_C * BLIS_DEFAULT_KC_C * SIZEOF_C ) ) || \
      ( ( BLIS_DEFAULT_MC_D * BLIS_DEFAULT_KC_D * SIZEOF_D ) < \
        ( BLIS_DEFAULT_MC_Z * BLIS_DEFAULT_KC_Z * SIZEOF_Z ) )    \
    )
  #error MC_D*KC_D must be >= that of MC*KC for all other datatypes.
#endif

#if ( \
      ( ( BLIS_DEFAULT_KC_D * BLIS_DEFAULT_NC_D * SIZEOF_D ) < \
        ( BLIS_DEFAULT_KC_S * BLIS_DEFAULT_NC_S * SIZEOF_S ) ) || \
      ( ( BLIS_DEFAULT_KC_D * BLIS_DEFAULT_NC_D * SIZEOF_D ) < \
        ( BLIS_DEFAULT_KC_C * BLIS_DEFAULT_NC_C * SIZEOF_C ) ) || \
      ( ( BLIS_DEFAULT_KC_D * BLIS_DEFAULT_NC_D * SIZEOF_D ) < \
        ( BLIS_DEFAULT_KC_Z * BLIS_DEFAULT_NC_Z * SIZEOF_Z ) )    \
    )
  #error KC_D*NC_D must be >= that of KC*NC for all other datatypes.
#endif

#if ( \
      ( ( BLIS_DEFAULT_MC_D * BLIS_DEFAULT_NC_D * SIZEOF_D ) < \
        ( BLIS_DEFAULT_MC_S * BLIS_DEFAULT_NC_S * SIZEOF_S ) ) || \
      ( ( BLIS_DEFAULT_MC_D * BLIS_DEFAULT_NC_D * SIZEOF_D ) < \
        ( BLIS_DEFAULT_MC_C * BLIS_DEFAULT_NC_C * SIZEOF_C ) ) || \
      ( ( BLIS_DEFAULT_MC_D * BLIS_DEFAULT_NC_D * SIZEOF_D ) < \
        ( BLIS_DEFAULT_MC_Z * BLIS_DEFAULT_NC_Z * SIZEOF_Z ) )    \
    )
  #error MC_D*NC_D must be >= that of MC*NC for all other datatypes.
#endif
*/


// -- Compute maximum cache blocksizes -----------------------------------------

#define BLIS_MAXIMUM_MC_S  ( BLIS_DEFAULT_MC_S + BLIS_EXTEND_MC_S )
#define BLIS_MAXIMUM_KC_S  ( BLIS_DEFAULT_KC_S + BLIS_EXTEND_KC_S )
#define BLIS_MAXIMUM_NC_S  ( BLIS_DEFAULT_NC_S + BLIS_EXTEND_NC_S )

#define BLIS_MAXIMUM_MC_D  ( BLIS_DEFAULT_MC_D + BLIS_EXTEND_MC_D )
#define BLIS_MAXIMUM_KC_D  ( BLIS_DEFAULT_KC_D + BLIS_EXTEND_KC_D )
#define BLIS_MAXIMUM_NC_D  ( BLIS_DEFAULT_NC_D + BLIS_EXTEND_NC_D )

#define BLIS_MAXIMUM_MC_C  ( BLIS_DEFAULT_MC_C + BLIS_EXTEND_MC_C )
#define BLIS_MAXIMUM_KC_C  ( BLIS_DEFAULT_KC_C + BLIS_EXTEND_KC_C )
#define BLIS_MAXIMUM_NC_C  ( BLIS_DEFAULT_NC_C + BLIS_EXTEND_NC_C )

#define BLIS_MAXIMUM_MC_Z  ( BLIS_DEFAULT_MC_Z + BLIS_EXTEND_MC_Z )
#define BLIS_MAXIMUM_KC_Z  ( BLIS_DEFAULT_KC_Z + BLIS_EXTEND_KC_Z )
#define BLIS_MAXIMUM_NC_Z  ( BLIS_DEFAULT_NC_Z + BLIS_EXTEND_NC_Z )


// -- Compute leading dim blocksizes used for packing --------------------------

#define BLIS_PACKDIM_MR_S  ( BLIS_DEFAULT_MR_S + BLIS_EXTEND_MR_S )
#define BLIS_PACKDIM_KR_S  ( BLIS_DEFAULT_KR_S + BLIS_EXTEND_KR_S )
#define BLIS_PACKDIM_NR_S  ( BLIS_DEFAULT_NR_S + BLIS_EXTEND_NR_S )

#define BLIS_PACKDIM_MR_D  ( BLIS_DEFAULT_MR_D + BLIS_EXTEND_MR_D )
#define BLIS_PACKDIM_KR_D  ( BLIS_DEFAULT_KR_D + BLIS_EXTEND_KR_D )
#define BLIS_PACKDIM_NR_D  ( BLIS_DEFAULT_NR_D + BLIS_EXTEND_NR_D )

#define BLIS_PACKDIM_MR_C  ( BLIS_DEFAULT_MR_C + BLIS_EXTEND_MR_C )
#define BLIS_PACKDIM_KR_C  ( BLIS_DEFAULT_KR_C + BLIS_EXTEND_KR_C )
#define BLIS_PACKDIM_NR_C  ( BLIS_DEFAULT_NR_C + BLIS_EXTEND_NR_C )

#define BLIS_PACKDIM_MR_Z  ( BLIS_DEFAULT_MR_Z + BLIS_EXTEND_MR_Z )
#define BLIS_PACKDIM_KR_Z  ( BLIS_DEFAULT_KR_Z + BLIS_EXTEND_KR_Z )
#define BLIS_PACKDIM_NR_Z  ( BLIS_DEFAULT_NR_Z + BLIS_EXTEND_NR_Z )


// -- Abbreiviated kernel blocksize macros -------------------------------------

// Here, we shorten the blocksizes defined in bli_kernel.h so that they can
// derived via the PASTEMAC macro.

// Default cache blocksizes

#define bli_smc      BLIS_DEFAULT_MC_S 
#define bli_skc      BLIS_DEFAULT_KC_S
#define bli_snc      BLIS_DEFAULT_NC_S

#define bli_dmc      BLIS_DEFAULT_MC_D 
#define bli_dkc      BLIS_DEFAULT_KC_D
#define bli_dnc      BLIS_DEFAULT_NC_D

#define bli_cmc      BLIS_DEFAULT_MC_C 
#define bli_ckc      BLIS_DEFAULT_KC_C
#define bli_cnc      BLIS_DEFAULT_NC_C

#define bli_zmc      BLIS_DEFAULT_MC_Z 
#define bli_zkc      BLIS_DEFAULT_KC_Z
#define bli_znc      BLIS_DEFAULT_NC_Z

// Maximum cache blocksizes

#define bli_smaxmc   BLIS_MAXIMUM_MC_S
#define bli_smaxkc   BLIS_MAXIMUM_KC_S
#define bli_smaxnc   BLIS_MAXIMUM_NC_S

#define bli_dmaxmc   BLIS_MAXIMUM_MC_D
#define bli_dmaxkc   BLIS_MAXIMUM_KC_D
#define bli_dmaxnc   BLIS_MAXIMUM_NC_D

#define bli_cmaxmc   BLIS_MAXIMUM_MC_C
#define bli_cmaxkc   BLIS_MAXIMUM_KC_C
#define bli_cmaxnc   BLIS_MAXIMUM_NC_C

#define bli_zmaxmc   BLIS_MAXIMUM_MC_Z
#define bli_zmaxkc   BLIS_MAXIMUM_KC_Z
#define bli_zmaxnc   BLIS_MAXIMUM_NC_Z

// Register blocksizes

#define bli_smr      BLIS_DEFAULT_MR_S 
#define bli_skr      BLIS_DEFAULT_KR_S
#define bli_snr      BLIS_DEFAULT_NR_S

#define bli_dmr      BLIS_DEFAULT_MR_D 
#define bli_dkr      BLIS_DEFAULT_KR_D
#define bli_dnr      BLIS_DEFAULT_NR_D

#define bli_cmr      BLIS_DEFAULT_MR_C 
#define bli_ckr      BLIS_DEFAULT_KR_C
#define bli_cnr      BLIS_DEFAULT_NR_C

#define bli_zmr      BLIS_DEFAULT_MR_Z 
#define bli_zkr      BLIS_DEFAULT_KR_Z
#define bli_znr      BLIS_DEFAULT_NR_Z

// Micro-panel packing register blocksizes

#define bli_spackmr  BLIS_PACKDIM_MR_S
#define bli_spackkr  BLIS_PACKDIM_KR_S
#define bli_spacknr  BLIS_PACKDIM_NR_S

#define bli_dpackmr  BLIS_PACKDIM_MR_D
#define bli_dpackkr  BLIS_PACKDIM_KR_D
#define bli_dpacknr  BLIS_PACKDIM_NR_D

#define bli_cpackmr  BLIS_PACKDIM_MR_C
#define bli_cpackkr  BLIS_PACKDIM_KR_C
#define bli_cpacknr  BLIS_PACKDIM_NR_C

#define bli_zpackmr  BLIS_PACKDIM_MR_Z
#define bli_zpackkr  BLIS_PACKDIM_KR_Z
#define bli_zpacknr  BLIS_PACKDIM_NR_Z

// Duplication factors

#define bli_sndup    BLIS_DEFAULT_NUM_DUPL_S
#define bli_dndup    BLIS_DEFAULT_NUM_DUPL_D
#define bli_cndup    BLIS_DEFAULT_NUM_DUPL_C
#define bli_zndup    BLIS_DEFAULT_NUM_DUPL_Z



#endif 
