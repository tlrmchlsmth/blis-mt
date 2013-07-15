#!/bin/bash +x
#qsub -n 1 -t 5 --mode=c1 --env XL_BG_SPREADLAYOUT=YES ./test_libblis.x
#echo "qsub -n 1 -t 5 --mode=c1 --env BG_THREADLAYOUT=2:BLIS_L2_NT=1:OMP_NUM_THREADS=4 $1"

TIME=45
LAYOUT=2
OMP_THREADS=64
#echo "qsub -n 1 -t $TIME --mode=c$N_RANKS --env BG_THREADLAYOUT=$LAYOUT:OMP_NUM_THREADS=$OMP_THREADS $1"
qsub -A PARTS -n 32 -t $TIME --mode=c1 --env BG_THREADLAYOUT=$LAYOUT:OMP_NUM_THREADS=$OMP_THREADS:BLIS_L1_NT=2:BLIS_L2_NT=32:BLIS_L3_NT=1:BLIS_L4_NT=1 $1
