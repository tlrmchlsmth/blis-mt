#!/bin/bash
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2013, The University of Texas
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name of The University of Texas nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

# Only include this block of code once.
ifndef MAKE_DEFS_MK_INCLUDED
MAKE_DEFS_MK_INCLUDED := yes



#
# --- Build definitions --------------------------------------------------------
#

# Variables corresponding to other configure-time options.
BLIS_ENABLE_VERBOSE_MAKE_OUTPUT := no
BLIS_ENABLE_STATIC_BUILD        := yes
BLIS_ENABLE_DYNAMIC_BUILD       := no



#
# --- Utility program definitions ----------------------------------------------
#

SH         := /bin/sh
MV         := mv
MKDIR      := mkdir -p
RM_F       := rm -f
RM_RF      := rm -rf
SYMLINK    := ln -sf
FIND       := find
XARGS      := xargs
RANLIB     := ranlib
INSTALL    := install -c

# Used to refresh CHANGELOG.
GIT        := git
GIT_LOG    := $(GIT) log --decorate



#
# --- Development tools definitions --------------------------------------------
#

# --- Determine the C compiler and related flags ---
<<<<<<< HEAD
CC           := mpixlc_r
# Enable IEEE Standard 1003.1-2004 (POSIX.1d). 
# NOTE: This is needed to enable posix_memalign().
CPPROCFLAGS  := -D_POSIX_C_SOURCE=200112L
CINCFLAGS    := -I/bgsys/drivers/ppcfloor -I/bgsys/drivers/ppcfloor/spi/include/kernel/cnk
CMISCFLAGS   := -qasm=gcc -qkeyword=asm -qthreaded -qsmp=omp # -qreport -qsource -qlistopt -qlist
CDBGFLAGS    := 
CWARNFLAGS   := -w
COPTFLAGS    := -O5
CVECFLAGS    := -qarch=qp -qtune=qp -qsimd=auto -qhot=level=2 -qprefetch -qunroll=yes 

# Aggregate all of the flags into two groups: one for optimizable code, and
# one for code that should not be optimized.
CFLAGS       := $(CINCFLAGS) $(CDBGFLAGS) $(COPTFLAGS) $(CVECFLAGS) $(CWARNFLAGS) $(CMISCFLAGS) $(CPPROCFLAGS)
CFLAGS_NOOPT := $(CINCFLAGS) $(CDBGFLAGS) $(CWARNFLAGS) $(CMISCFLAGS) $(CPPROCFLAGS)

# --- Determine the archiver and related flags ---
AR           := ar
ARFLAGS      := cru

# --- Determine the linker and related flags ---
LINKER       := $(CC)
LDFLAGS      := -L/bgsys/drivers/ppcfloor/spi/lib -lSPI -lSPI_cnk -qthreaded -qsmp=omp
=======
CC             := /bgsys/drivers/ppcfloor/comm/gcc.legacy/bin/mpixlc_r
# Enable IEEE Standard 1003.1-2004 (POSIX.1d). 
# NOTE: This is needed to enable posix_memalign().
CPPROCFLAGS    := -D_POSIX_C_SOURCE=200112L \
                  -I/bgsys/drivers/ppcfloor -I/bgsys/drivers/ppcfloor/spi/include/kernel/cnk
CMISCFLAGS     := -qthreaded -qsmp=omp -qasm=gcc -qkeyword=asm # -qreport -qsource -qlistopt -qlist
CDBGFLAGS      :=
CWARNFLAGS     := -w
COPTFLAGS      := -O3
CKOPTFLAGS     := $(COPTFLAGS)
CVECFLAGS      := -qarch=qp -qtune=qp -qsimd=auto -qhot=level=1 -qprefetch -qunroll=yes -qnoipa

# Aggregate all of the flags into multiple groups: one for standard
# compilation, and one for each of the supported "special" compilation
# modes.
CFLAGS         := $(CDBGFLAGS) $(COPTFLAGS)  $(CVECFLAGS) $(CWARNFLAGS) $(CMISCFLAGS) $(CPPROCFLAGS)
CFLAGS_KERNELS := $(CDBGFLAGS) $(CKOPTFLAGS) $(CVECFLAGS) $(CWARNFLAGS) $(CMISCFLAGS) $(CPPROCFLAGS)
CFLAGS_NOOPT   := $(CDBGFLAGS)                            $(CWARNFLAGS) $(CMISCFLAGS) $(CPPROCFLAGS)

# --- Determine the archiver and related flags ---
AR             := ar
ARFLAGS        := cru

# --- Determine the linker and related flags ---
LINKER         := $(CC)
LDFLAGS        := -L/bgsys/drivers/ppcfloor/spi/lib -lSPI -lSPI_cnk -qthreaded -qsmp=omp
>>>>>>> a091a219bda55e56817acd4930c2aa4472e53ba5



# end of ifndef MAKE_DEFS_MK_INCLUDED conditional block
endif
