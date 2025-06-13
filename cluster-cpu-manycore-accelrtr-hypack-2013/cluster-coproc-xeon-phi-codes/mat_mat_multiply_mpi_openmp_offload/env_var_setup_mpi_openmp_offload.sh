#
#		C-DAC Tech Workshop : hyPACK-2013
#                        October 15-18, 2013
#
#   Created           :  August-2013
#
#   E-mail            :  hpcfte@cdac.in     
#
#!/bin/bash

#set environment variables
export MIC_ENV_PREFIX=PHI
export PHI_OMP_NUM_THREADS=236
export PHI_KMP_AFFINITY=granularity=fine,compact
export OMP_NUM_THREADS=236
export KMP_AFFINITY=granularity=fine,compact

#compilation
make -f Makefile.OFFLOAD clean 
make -f Makefile.OFFLOAD

#execution
mpirun -machinefile mpi_hosts -np 8 ./run 16000 236 4

#unset the env variables
unset MIC_ENV_PREFIX
unset PHI_OMP_NUM_THREADS
unset PHI_KMP_AFFINITY
unset OMP_NUM_THREADS
unset KMP_AFFINITY
make -f Makefile.OFFLOAD clean 
