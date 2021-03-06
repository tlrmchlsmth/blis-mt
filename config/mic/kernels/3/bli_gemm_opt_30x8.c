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
   AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
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
#include <assert.h>

void bli_sgemm_opt_30x8(
                    dim_t     k,	    float*    alpha,
                    float*    a,
                    float*    b,
                    float*    beta,
                    float*    c, inc_t rs_c, inc_t cs_c,
                    float* a_next,
                    float* b_next,
                    dim_t thread_id
                  )
{
}

#define UPDATE_C_ROW_SCATTERED(REG1, NUM, BASE_DEST) \
{ \
        __asm kmov k3, ebx \
        __asm GATHER##NUM: \
            __asm vgatherdpd zmm31{k3}, [BASE_DEST + zmm30 * 8] \
            __asm jknzd k3, GATHER##NUM \
        \
        __asm vmulpd REG1, REG1, 0[r12]{1to8} /*scale by alpha*/ \
        __asm vfmadd132pd zmm31, REG1, 0[r13]{1to8} /*scale by beta, add in result*/\
        __asm kmov k3, ebx \
        \
        __asm SCATTER##NUM: \
            __asm vscatterdpd [BASE_DEST + zmm30 * 8]{k3}, zmm31 \
            __asm jknzd k3, SCATTER##NUM \
        __asm add BASE_DEST, r11 \
}


#define ONE_ITER_PA(C_ADDR) \
{\
        __asm vbroadcastf64x4   zmm30, 0[r15] \
        __asm vmovapd zmm31, 0[rbx]           \
\
        __asm vfmadd231pd zmm4, zmm31,  4*8[r15]{1to8} \
        __asm add r15, 32*8 \
        __asm vfmadd231pd zmm5, zmm31,  -27*8[r15]{1to8} \
        __asm vfmadd231pd zmm6, zmm31,  -26*8[r15]{1to8} \
        __asm add r14, 256 \
        __asm vfmadd231pd zmm7, zmm31,  -25*8[r15]{1to8} \
        \
        __asm vprefetch0 0*0[r15] \
        __asm vfmadd231pd zmm0, zmm31, zmm30{aaaa} \
        __asm vprefetch0 8*8[r15] \
        __asm vfmadd231pd zmm1, zmm31, zmm30{bbbb} \
        __asm vprefetch0 8*8[rbx] \
        __asm vfmadd231pd zmm2, zmm31, zmm30{cccc} \
        __asm vprefetch0 8*16[r15] \
        __asm vfmadd231pd zmm3, zmm31, zmm30{dddd} \
        __asm vprefetch0 8*24[r15] \
\
        __asm vfmadd231pd zmm8, zmm31,  -24*8[r15]{1to8} \
        __asm add rbx, 8*8 \
        __asm vfmadd231pd zmm9, zmm31,  -23*8[r15]{1to8} \
        __asm vfmadd231pd zmm10, zmm31, -22*8[r15]{1to8} \
        __asm vfmadd231pd zmm11, zmm31, -21*8[r15]{1to8} \
\
        __asm vprefetch1 0[r13] \
        __asm vfmadd231pd zmm12, zmm31, -20*8[r15]{1to8} \
        __asm add r13, 8*8 \
        __asm vfmadd231pd zmm13, zmm31, -19*8[r15]{1to8} \
        __asm dec rsi \
        __asm vfmadd231pd zmm14, zmm31, -18*8[r15]{1to8} \
\
        __asm vprefetch1 0[r14] \
        __asm vfmadd231pd zmm15, zmm31, -17*8[r15]{1to8} \
        __asm vfmadd231pd zmm16, zmm31, -16*8[r15]{1to8} \
        __asm vfmadd231pd zmm17, zmm31, -15*8[r15]{1to8} \
\
        __asm vprefetch1 64[r14] \
        __asm vfmadd231pd zmm18, zmm31, -14*8[r15]{1to8} \
        __asm vfmadd231pd zmm19, zmm31, -13*8[r15]{1to8} \
        __asm vfmadd231pd zmm20, zmm31, -12*8[r15]{1to8} \
\
        __asm vprefetch1 2*64[r14] \
        __asm vfmadd231pd zmm21, zmm31, -11*8[r15]{1to8} \
        __asm vfmadd231pd zmm22, zmm31, -10*8[r15]{1to8} \
        __asm vfmadd231pd zmm23, zmm31, -9*8[r15]{1to8} \
\
        __asm vprefetch1 3*64[r14] \
        __asm vfmadd231pd zmm24, zmm31, -8*8[r15]{1to8} \
        __asm vfmadd231pd zmm25, zmm31, -7*8[r15]{1to8} \
        __asm vfmadd231pd zmm26, zmm31, -6*8[r15]{1to8} \
\
        __asm vfmadd231pd zmm27, zmm31, -5*8[r15]{1to8} \
        __asm vfmadd231pd zmm28, zmm31, -4*8[r15]{1to8} \
        __asm vfmadd231pd zmm29, zmm31, -3*8[r15]{1to8} \
        \
}
#define ONE_ITER(C_ADDR) \
{\
        __asm vbroadcastf64x4   zmm30, 0[r15] \
        __asm vmovapd zmm31, 0[rbx]           \
\
        __asm vfmadd231pd zmm4, zmm31,  4*8[r15]{1to8} \
        __asm add r15, 32*8 \
        __asm vfmadd231pd zmm5, zmm31,  -27*8[r15]{1to8} \
        __asm vfmadd231pd zmm6, zmm31,  -26*8[r15]{1to8} \
        __asm add r14, 256 \
        __asm vfmadd231pd zmm7, zmm31,  -25*8[r15]{1to8} \
        \
        __asm vprefetch0 0*0[r15] \
        __asm vfmadd231pd zmm0, zmm31, zmm30{aaaa} \
        __asm vprefetch0 8*8[r15] \
        __asm vfmadd231pd zmm1, zmm31, zmm30{bbbb} \
        __asm vprefetch0 8*8[rbx] \
        __asm vfmadd231pd zmm2, zmm31, zmm30{cccc} \
        __asm vprefetch0 8*16[r15] \
        __asm vfmadd231pd zmm3, zmm31, zmm30{dddd} \
        __asm vprefetch0 8*24[r15] \
\
        __asm vfmadd231pd zmm8, zmm31,  -24*8[r15]{1to8} \
        __asm add rbx, 8*8 \
        __asm vfmadd231pd zmm9, zmm31,  -23*8[r15]{1to8} \
        __asm vfmadd231pd zmm10, zmm31, -22*8[r15]{1to8} \
        __asm vfmadd231pd zmm11, zmm31, -21*8[r15]{1to8} \
\
        __asm vprefetch1 0[r13] \
        __asm vfmadd231pd zmm12, zmm31, -20*8[r15]{1to8} \
        __asm add r13, 8*8 \
        __asm vfmadd231pd zmm13, zmm31, -19*8[r15]{1to8} \
        __asm dec rsi \
        __asm vfmadd231pd zmm14, zmm31, -18*8[r15]{1to8} \
\
        /*__asm vprefetch1 0[r14] */\
        __asm vfmadd231pd zmm15, zmm31, -17*8[r15]{1to8} \
        __asm vfmadd231pd zmm16, zmm31, -16*8[r15]{1to8} \
        __asm vfmadd231pd zmm17, zmm31, -15*8[r15]{1to8} \
\
        /*__asm vprefetch1 64[r14] */\
        __asm vfmadd231pd zmm18, zmm31, -14*8[r15]{1to8} \
        __asm vfmadd231pd zmm19, zmm31, -13*8[r15]{1to8} \
        __asm vfmadd231pd zmm20, zmm31, -12*8[r15]{1to8} \
\
        /*__asm vprefetch1 2*64[r14] */\
        __asm vfmadd231pd zmm21, zmm31, -11*8[r15]{1to8} \
        __asm vfmadd231pd zmm22, zmm31, -10*8[r15]{1to8} \
        __asm vfmadd231pd zmm23, zmm31, -9*8[r15]{1to8} \
\
        /*__asm vprefetch1 3*64[r14] */\
        __asm vfmadd231pd zmm24, zmm31, -8*8[r15]{1to8} \
        __asm vfmadd231pd zmm25, zmm31, -7*8[r15]{1to8} \
        __asm vfmadd231pd zmm26, zmm31, -6*8[r15]{1to8} \
\
        __asm vfmadd231pd zmm27, zmm31, -5*8[r15]{1to8} \
        __asm vfmadd231pd zmm28, zmm31, -4*8[r15]{1to8} \
        __asm vfmadd231pd zmm29, zmm31, -3*8[r15]{1to8} \
        \
}

#define ONE_ITER_PC_L1(C_ADDR) \
{\
        __asm vbroadcastf64x4   zmm30, 0[r15] \
        __asm vmovapd zmm31, 0[rbx]           \
\
        __asm vfmadd231pd zmm4, zmm31,  4*8[r15]{1to8} \
        __asm add r15, 32*8 \
        __asm vfmadd231pd zmm5, zmm31,  -27*8[r15]{1to8} \
        __asm vfmadd231pd zmm6, zmm31,  -26*8[r15]{1to8} \
        __asm add r14, 256 \
        __asm vfmadd231pd zmm7, zmm31,  -25*8[r15]{1to8} \
        \
        __asm vprefetch0 8*8[r15] \
        __asm vfmadd231pd zmm0, zmm31, zmm30{aaaa} \
        __asm vprefetch0 0*0[r15] \
        __asm vfmadd231pd zmm1, zmm31, zmm30{bbbb} \
        __asm vprefetch0 8*8[rbx] \
        __asm vfmadd231pd zmm2, zmm31, zmm30{cccc} \
        __asm vprefetch0 8*16[r15] \
        __asm vfmadd231pd zmm3, zmm31, zmm30{dddd} \
        __asm vprefetch0 8*24[r15] \
\
        __asm vfmadd231pd zmm8, zmm31,  -24*8[r15]{1to8} \
        __asm vprefetche0 0[C_ADDR] \
        __asm vfmadd231pd zmm9, zmm31,  -23*8[r15]{1to8} \
        __asm add rbx, 8*8 \
        __asm vfmadd231pd zmm10, zmm31, -22*8[r15]{1to8} \
        __asm vprefetch1 0[r13] \
        __asm vfmadd231pd zmm11, zmm31, -21*8[r15]{1to8} \
        __asm add r13, 8*8 \
\
        __asm vfmadd231pd zmm12, zmm31, -20*8[r15]{1to8} \
        __asm add C_ADDR, r11 \
        __asm vfmadd231pd zmm13, zmm31, -19*8[r15]{1to8} \
        __asm vprefetche0 0[C_ADDR] \
        __asm vfmadd231pd zmm14, zmm31, -18*8[r15]{1to8} \
\
        __asm vprefetch1 0[r14] \
        __asm vfmadd231pd zmm15, zmm31, -17*8[r15]{1to8} \
        __asm add C_ADDR, r11 \
        __asm vfmadd231pd zmm16, zmm31, -16*8[r15]{1to8} \
        __asm vfmadd231pd zmm17, zmm31, -15*8[r15]{1to8} \
\
        __asm vprefetch1 64[r14] \
        __asm vfmadd231pd zmm18, zmm31, -14*8[r15]{1to8} \
        __asm vprefetche0 0[C_ADDR] \
        __asm vfmadd231pd zmm19, zmm31, -13*8[r15]{1to8} \
        __asm add C_ADDR, r11 \
        __asm vfmadd231pd zmm20, zmm31, -12*8[r15]{1to8} \
\
        __asm dec rdx \
        __asm vfmadd231pd zmm21, zmm31, -11*8[r15]{1to8} \
        __asm vprefetch1 2*64[r14] \
        __asm vfmadd231pd zmm22, zmm31, -10*8[r15]{1to8} \
        __asm vfmadd231pd zmm23, zmm31, -9*8[r15]{1to8} \
\
        __asm vprefetch1 3*64[r14] \
        __asm vfmadd231pd zmm24, zmm31, -8*8[r15]{1to8} \
        __asm vfmadd231pd zmm25, zmm31, -7*8[r15]{1to8} \
        __asm vfmadd231pd zmm26, zmm31, -6*8[r15]{1to8} \
\
        __asm vfmadd231pd zmm27, zmm31, -5*8[r15]{1to8} \
        __asm vfmadd231pd zmm28, zmm31, -4*8[r15]{1to8} \
        __asm vfmadd231pd zmm29, zmm31, -3*8[r15]{1to8} \
        \
}


#define ONE_ITER_PC_L2(C_ADDR) \
{\
        __asm vbroadcastf64x4   zmm30, 0[r15] \
        __asm vmovapd zmm31, 0[rbx]           \
\
        __asm vfmadd231pd zmm4, zmm31,  4*8[r15]{1to8} \
        __asm add r15, 32*8 \
        __asm vfmadd231pd zmm5, zmm31,  -27*8[r15]{1to8} \
        __asm vfmadd231pd zmm6, zmm31,  -26*8[r15]{1to8} \
        __asm add r14, 256 \
        __asm vfmadd231pd zmm7, zmm31,  -25*8[r15]{1to8} \
        \
        __asm vprefetch0 8*16[r15] \
        __asm vfmadd231pd zmm0, zmm31, zmm30{aaaa} \
        __asm vprefetch0 8*8[rbx] \
        __asm vfmadd231pd zmm1, zmm31, zmm30{bbbb} \
        __asm vprefetch0 8*8[r15] \
        __asm vfmadd231pd zmm2, zmm31, zmm30{cccc} \
        __asm vprefetch0 0*0[r15] \
        __asm vfmadd231pd zmm3, zmm31, zmm30{dddd} \
        __asm vprefetch0 8*24[r15] \
\
        __asm vprefetch1 0[r13] \
        __asm vfmadd231pd zmm8, zmm31,  -24*8[r15]{1to8} \
        __asm add rbx, 8*8 \
        __asm vfmadd231pd zmm9, zmm31,  -23*8[r15]{1to8} \
        __asm add r13, 8*8 \
        __asm vfmadd231pd zmm10, zmm31, -22*8[r15]{1to8} \
        __asm vfmadd231pd zmm11, zmm31, -21*8[r15]{1to8} \
\
        __asm vprefetche1 0[C_ADDR] \
        __asm vfmadd231pd zmm12, zmm31, -20*8[r15]{1to8} \
        __asm add C_ADDR, r11 \
        __asm vfmadd231pd zmm13, zmm31, -19*8[r15]{1to8} \
        __asm dec rdx \
        __asm vfmadd231pd zmm14, zmm31, -18*8[r15]{1to8} \
\
        __asm vfmadd231pd zmm15, zmm31, -17*8[r15]{1to8} \
        __asm vfmadd231pd zmm16, zmm31, -16*8[r15]{1to8} \
        __asm vfmadd231pd zmm17, zmm31, -15*8[r15]{1to8} \
\
        __asm vprefetch1 0[r14] \
        __asm vfmadd231pd zmm18, zmm31, -14*8[r15]{1to8} \
        __asm vfmadd231pd zmm19, zmm31, -13*8[r15]{1to8} \
        __asm vfmadd231pd zmm20, zmm31, -12*8[r15]{1to8} \
\
        __asm vprefetch1 64[r14] \
        __asm vfmadd231pd zmm21, zmm31, -11*8[r15]{1to8} \
        __asm vfmadd231pd zmm22, zmm31, -10*8[r15]{1to8} \
        __asm vfmadd231pd zmm23, zmm31, -9*8[r15]{1to8} \
\
        __asm vprefetch1 2*64[r14] \
        __asm vfmadd231pd zmm24, zmm31, -8*8[r15]{1to8} \
        __asm vfmadd231pd zmm25, zmm31, -7*8[r15]{1to8} \
        __asm vfmadd231pd zmm26, zmm31, -6*8[r15]{1to8} \
\
        __asm vprefetch1 3*64[r14] \
        __asm vfmadd231pd zmm27, zmm31, -5*8[r15]{1to8} \
        __asm vfmadd231pd zmm28, zmm31, -4*8[r15]{1to8} \
        __asm vfmadd231pd zmm29, zmm31, -3*8[r15]{1to8} \
}

//This is an array used for the scattter/gather instructions.
int offsets[16] __attribute__((aligned(0x1000))) = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};


//#define MONITORS
//#define LOOPMON
void bli_dgemm_opt_30x8(
                    dim_t     k,	    double*   alpha,
                    double*   a,
                    double*   b,
                    double*   beta,
                    double*   c, inc_t rs_c, inc_t cs_c,
                    double* a_next,
                    double* b_next,
                    dim_t thread_id
                  )
{

    int * offsetPtr = &offsets[0];

#ifdef MONITORS
    int toph, topl, both, botl, midl, midh, mid2l, mid2h;
#endif
#ifdef LOOPMON
    int tlooph, tloopl, blooph, bloopl;
#endif
    dim_t k_iter, k_left;
    k_iter = k; //BLIS can optionally pad k to multiples of 8 so we don't have to deal with a cleanup loop.
    
    //printf("%u\t%u\n", ((unsigned int) a) % 64, ((unsigned int) b) % 64);
    //return;   
    
    __asm
    {
#ifdef MONITORS
        rdtsc
        mov topl, eax
        mov toph, edx 
#endif
        vpxord  zmm0,  zmm0, zmm0
        vmovaps zmm1,  zmm0  //clear out registers
        mov r12, alpha
        vmovaps zmm2,  zmm0 
        mov rsi, k_iter    //loop index
        vmovaps zmm3,  zmm0 

        mov r11, rs_c           //load row stride
        vmovaps zmm4,  zmm0 
        sal r11, 3              //scale row stride
        vmovaps zmm5,  zmm0 
        mov r15, a              //load address of a
        vmovaps zmm6,  zmm0 
        mov rbx, b              //load address of b
        vmovaps zmm7,  zmm0 

        vmovaps zmm8,  zmm0 
        lea r10, [r11 + 2*r11 + 0] //r10 has 3 * r11
        vmovaps zmm9,  zmm0
        vmovaps zmm10, zmm0 
        mov rdi, r11    
        vmovaps zmm11, zmm0 
        sal rdi, 2              //rdi has 4*r11

        vmovaps zmm12, zmm0 
        mov rcx, c              //load address of c for prefetching
        vmovaps zmm13, zmm0 
        mov r9, c               //load address of c for update
        vmovaps zmm14, zmm0 
        mov rdx, k 
        vmovaps zmm15, zmm0 
        mov r13, b_next

        vmovaps zmm16, zmm0
        mov r14, a_next
        vmovaps zmm17, zmm0
        add r14, 256
        vmovaps zmm18, zmm0 
        sub rdx, 40             //Check if we have over 40 operations to do.
        vmovaps zmm19, zmm0 
        mov rdx, 30
        vmovaps zmm20, zmm0 
        vmovaps zmm21, zmm0 
        vmovaps zmm22, zmm0 

        vmovaps zmm23, zmm0 
        vmovaps zmm24, zmm0 
        vmovaps zmm25, zmm0 
        vmovaps zmm26, zmm0 
        vmovaps zmm27, zmm0 
        vmovaps zmm28, zmm0 
        vmovaps zmm29, zmm0 

#ifdef MONITORS
        rdtsc
        mov midl, eax
        mov midh, edx 
#endif
        jle CONSIDER_UNDER_40
        sub rsi, 40


        //First 30 iterations.
        LOOPREFECHCL2:
            ONE_ITER_PC_L2(rcx)
        jne LOOPREFECHCL2
        mov rcx, c

        //Main Loop.
        LOOPKITER:      
            ONE_ITER_PA(rcx)
        jne LOOPKITER

        //Last 10 iterations
        mov rdx, 10
        LOOPREFETCHCL1:
            ONE_ITER_PC_L1(rcx)
        jne LOOPREFETCHCL1

        jmp POSTACCUM

        //Alternate main loop, with no prefetching of C
        //Used when <= 40 iterations
        CONSIDER_UNDER_40:
        mov rsi, k_iter
        LOOP_UNDER_40:
            ONE_ITER(rcx)
        jne LOOP_UNDER_40



        POSTACCUM:
#ifdef MONITORS
        rdtsc
        mov mid2l, eax
        mov mid2h, edx
#endif
        mov r14, cs_c
        dec r14
        jne SCATTEREDUPDATE

        mov r14, beta
        vbroadcastsd zmm31, 0[r14] 

        vmulpd zmm0, zmm0, 0[r12]{1to8}
        vmulpd zmm1, zmm1, 0[r12]{1to8}
        vmulpd zmm2, zmm2, 0[r12]{1to8}
        vmulpd zmm3, zmm3, 0[r12]{1to8}
        vfmadd231pd zmm0, zmm31, [r9+0]
        vfmadd231pd zmm1, zmm31, [r9+r11+0]
        vfmadd231pd zmm2, zmm31, [r9+2*r11+0]
        vfmadd231pd zmm3, zmm31, [r9+r10+0]
        vmovnrngoapd [r9+0], zmm0
        vmovnrngoapd [r9+r11+0], zmm1
        vmovnrngoapd [r9+2*r11+0], zmm2
        vmovnrngoapd [r9+r10+0], zmm3
        add r9, rdi

        vmulpd zmm4, zmm4, 0[r12]{1to8}
        vmulpd zmm5, zmm5, 0[r12]{1to8}
        vmulpd zmm6, zmm6, 0[r12]{1to8}
        vmulpd zmm7, zmm7, 0[r12]{1to8}
        vfmadd231pd zmm4, zmm31, [r9+0]
        vfmadd231pd zmm5, zmm31, [r9+r11+0]
        vfmadd231pd zmm6, zmm31, [r9+2*r11+0]
        vfmadd231pd zmm7, zmm31, [r9+r10+0]
        vmovnrngoapd [r9+0], zmm4
        vmovnrngoapd [r9+r11+0], zmm5
        vmovnrngoapd [r9+2*r11+0], zmm6
        vmovnrngoapd [r9+r10+0], zmm7
        add r9, rdi

        vmulpd zmm8, zmm8, 0[r12]{1to8}
        vmulpd zmm9, zmm9, 0[r12]{1to8}
        vmulpd zmm10, zmm10, 0[r12]{1to8}
        vmulpd zmm11, zmm11, 0[r12]{1to8}
        vfmadd231pd zmm8, zmm31, [r9+0]
        vfmadd231pd zmm9, zmm31, [r9+r11+0]
        vfmadd231pd zmm10, zmm31, [r9+2*r11+0]
        vfmadd231pd zmm11, zmm31, [r9+r10+0]
        vmovnrngoapd [r9+0], zmm8
        vmovnrngoapd [r9+r11+0], zmm9
        vmovnrngoapd [r9+2*r11+0], zmm10
        vmovnrngoapd [r9+r10+0], zmm11
        add r9, rdi

        vmulpd zmm12, zmm12, 0[r12]{1to8}
        vmulpd zmm13, zmm13, 0[r12]{1to8}
        vmulpd zmm14, zmm14, 0[r12]{1to8}
        vmulpd zmm15, zmm15, 0[r12]{1to8}
        vfmadd231pd zmm12, zmm31, [r9+0]
        vfmadd231pd zmm13, zmm31, [r9+r11+0]
        vfmadd231pd zmm14, zmm31, [r9+2*r11+0]
        vfmadd231pd zmm15, zmm31, [r9+r10+0]
        vmovnrngoapd [r9+0], zmm12
        vmovnrngoapd [r9+r11+0], zmm13
        vmovnrngoapd [r9+2*r11+0], zmm14
        vmovnrngoapd [r9+r10+0], zmm15
        add r9, rdi
        
        vmulpd zmm16, zmm16, 0[r12]{1to8}
        vmulpd zmm17, zmm17, 0[r12]{1to8}
        vmulpd zmm18, zmm18, 0[r12]{1to8}
        vmulpd zmm19, zmm19, 0[r12]{1to8}
        vfmadd231pd zmm16, zmm31, [r9+0]
        vfmadd231pd zmm17, zmm31, [r9+r11+0]
        vfmadd231pd zmm18, zmm31, [r9+2*r11+0]
        vfmadd231pd zmm19, zmm31, [r9+r10+0]
        vmovnrngoapd [r9+0], zmm16
        vmovnrngoapd [r9+r11+0], zmm17
        vmovnrngoapd [r9+2*r11+0], zmm18
        vmovnrngoapd [r9+r10+0], zmm19
        add r9, rdi

        vmulpd zmm20, zmm20, 0[r12]{1to8}
        vmulpd zmm21, zmm21, 0[r12]{1to8}
        vmulpd zmm22, zmm22, 0[r12]{1to8}
        vmulpd zmm23, zmm23, 0[r12]{1to8}
        vfmadd231pd zmm20, zmm31, [r9+0]
        vfmadd231pd zmm21, zmm31, [r9+r11+0]
        vfmadd231pd zmm22, zmm31, [r9+2*r11+0]
        vfmadd231pd zmm23, zmm31, [r9+r10+0]
        vmovnrngoapd [r9+0], zmm20
        vmovnrngoapd [r9+r11+0], zmm21
        vmovnrngoapd [r9+2*r11+0], zmm22
        vmovnrngoapd [r9+r10+0], zmm23
        add r9, rdi

        vmulpd zmm24, zmm24, 0[r12]{1to8}
        vmulpd zmm25, zmm25, 0[r12]{1to8}
        vmulpd zmm26, zmm26, 0[r12]{1to8}
        vmulpd zmm27, zmm27, 0[r12]{1to8}
        vfmadd231pd zmm24, zmm31, [r9+0]
        vfmadd231pd zmm25, zmm31, [r9+r11+0]
        vfmadd231pd zmm26, zmm31, [r9+2*r11+0]
        vfmadd231pd zmm27, zmm31, [r9+r10+0]
        vmovnrngoapd [r9+0], zmm24
        vmovnrngoapd [r9+r11+0], zmm25
        vmovnrngoapd [r9+2*r11+0], zmm26
        vmovnrngoapd [r9+r10+0], zmm27
        add r9, rdi

        vmulpd zmm28, zmm28, 0[r12]{1to8}
        vmulpd zmm29, zmm29, 0[r12]{1to8}
        vfmadd231pd zmm28, zmm31, [r9+0]
        vfmadd231pd zmm29, zmm31, [r9+r11+0]
        vmovnrngoapd [r9+0], zmm28
        vmovnrngoapd [r9+r11+0], zmm29
        
        jmp END
        
        SCATTEREDUPDATE:
        mov r10, offsetPtr 
        vmovapd zmm31, 0[r10] 
        vpbroadcastd zmm30, cs_c 
        mov r13, beta
        vpmulld zmm30, zmm31, zmm30 
       
        mov ebx, 255 
        UPDATE_C_ROW_SCATTERED(zmm0, 0, r9) 
        UPDATE_C_ROW_SCATTERED(zmm1, 1, r9) 
        UPDATE_C_ROW_SCATTERED(zmm2, 2, r9) 
        UPDATE_C_ROW_SCATTERED(zmm3, 3, r9) 
        UPDATE_C_ROW_SCATTERED(zmm4, 4, r9) 
        UPDATE_C_ROW_SCATTERED(zmm5, 5, r9) 
        UPDATE_C_ROW_SCATTERED(zmm6, 6, r9) 
        UPDATE_C_ROW_SCATTERED(zmm7, 7, r9) 
        UPDATE_C_ROW_SCATTERED(zmm8, 8, r9) 
        UPDATE_C_ROW_SCATTERED(zmm9, 9, r9) 
        UPDATE_C_ROW_SCATTERED(zmm10, 10, r9) 
        UPDATE_C_ROW_SCATTERED(zmm11, 11, r9) 
        UPDATE_C_ROW_SCATTERED(zmm12, 12, r9) 
        UPDATE_C_ROW_SCATTERED(zmm13, 13, r9) 
        UPDATE_C_ROW_SCATTERED(zmm14, 14, r9) 
        UPDATE_C_ROW_SCATTERED(zmm15, 15, r9) 
        UPDATE_C_ROW_SCATTERED(zmm16, 16, r9) 
        UPDATE_C_ROW_SCATTERED(zmm17, 17, r9) 
        UPDATE_C_ROW_SCATTERED(zmm18, 18, r9) 
        UPDATE_C_ROW_SCATTERED(zmm19, 19, r9) 
        UPDATE_C_ROW_SCATTERED(zmm20, 20, r9) 
        UPDATE_C_ROW_SCATTERED(zmm21, 21, r9) 
        UPDATE_C_ROW_SCATTERED(zmm22, 22, r9) 
        UPDATE_C_ROW_SCATTERED(zmm23, 23, r9) 
        UPDATE_C_ROW_SCATTERED(zmm24, 24, r9) 
        UPDATE_C_ROW_SCATTERED(zmm25, 25, r9) 
        UPDATE_C_ROW_SCATTERED(zmm26, 26, r9) 
        UPDATE_C_ROW_SCATTERED(zmm27, 27, r9) 
        UPDATE_C_ROW_SCATTERED(zmm28, 28, r9) 
        UPDATE_C_ROW_SCATTERED(zmm29, 29, r9)

        END:
#ifdef MONITORS
        rdtsc
        mov botl, eax
        mov both, edx
#endif
    }

#ifdef LOOPMON
    printf("looptime = \t%d\n", bloopl - tloopl);
#endif
#ifdef MONITORS
    dim_t top = ((dim_t)toph << 32) | topl;
    dim_t mid = ((dim_t)midh << 32) | midl;
    dim_t mid2 = ((dim_t)mid2h << 32) | mid2l;
    dim_t bot = ((dim_t)both << 32) | botl;
    printf("setup =\t%u\tmain loop =\t%u\tcleanup=\t%u\ttotal=\t%u\n", mid - top, mid2 - mid, bot - mid2, bot - top);
#endif
}



void bli_cgemm_opt_30x8(
                    dim_t     k,	    scomplex* alpha,  
                    scomplex* a,
                    scomplex* b,
                    scomplex* beta,
                    scomplex* c, inc_t rs_c, inc_t cs_c,
                    scomplex* a_next,
                    scomplex* b_next,
                    dim_t thread_id
                  )
{
}


void bli_zgemm_opt_30x8(
                    dim_t     k,	    dcomplex* alpha,  
                    dcomplex* a,
                    dcomplex* b,
                    dcomplex* beta,
                    dcomplex* c, inc_t rs_c, inc_t cs_c,
                    dcomplex* a_next,
                    dcomplex* b_next,
                    dim_t thread_id
                  )
{
}
