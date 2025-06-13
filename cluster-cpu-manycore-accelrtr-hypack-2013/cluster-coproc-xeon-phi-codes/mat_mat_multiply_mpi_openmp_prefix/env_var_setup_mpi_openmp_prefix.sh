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
export I_MPI_MIC=enable
export I_MPI_FALLBACK=1
export LD_LIBRARY_PATH=/opt/intel/lib/mic/:/opt/intel/mkl/lib/mic/:${LD_LIBRARY_PATH}
export I_MPI_MIC_PREFIX=./MIC/

#compilation
make -f Makefile.NATIVE clean
make -f Makefile.No-OFFLOAD clean 
make -f Makefile.NATIVE
make -f Makefile.No-OFFLOAD

#execution
mpirun -machinefile mpi_hosts -np 24 ./run 16000 236 4

#unset env variables
unset MIC_ENV_PREFIX
unset PHI_OMP_NUM_THREADS
unset PHI_KMP_AFFINITY
unset OMP_NUM_THREADS
unset KMP_AFFINITY
unset I_MPI_MIC
unset I_MPI_FALLBACK
unset LD_LIBRARY_PATH
unset I_MPI_MIC_PREFIX
make -f Makefile.NATIVE clean
make -f Makefile.No-OFFLOAD clean 
