#!/bin/bash

# set environment variables 
export MKL_NUM_THREADS=236 
export KMP_AFFINITY=granularity=fine,compact
export MIC_ENV_PREFIX=PHI 
export PHI_MKL_NUM_THREADS=236
export PHI_KMP_AFFINITY=granularity=fine,compact

# compilation of source code
make -f Makefile.OFFLOAD clean
make -f Makefile.OFFLOAD

#execution of source code
./run 0 0 16000

#unset env variables
unset MKL_NUM_THREADS
unset KMP_AFFINITY
unset MIC_ENV_PREFIX
unset PHI_MKL_NUM_THREADS
unset PHI_KMP_AFFINITY
make -f Makefile.OFFLOAD clean
