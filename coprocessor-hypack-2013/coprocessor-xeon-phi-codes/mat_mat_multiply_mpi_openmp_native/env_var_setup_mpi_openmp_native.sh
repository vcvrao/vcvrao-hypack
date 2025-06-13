
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
export I_MPI_MIC=enable
export I_MPI_FALLBACK=1
export LD_LIBRARY_PATH=/opt/intel/lib/mic/:/opt/intel/mkl/lib/mic/:${LD_LIBRARY_PATH}

#compilation
make -f Makefile_mat_mat_mpi_openmp.NATIVE clean
make -f Makefile_mat_mat_mpi_openmp.NATIVE

#execution
mpirun -machinefile  mpi_hosts_xeon_phi_native -np 2 ./run 4096 236 4

#unset env variables
unset MIC_ENV_PREFIX
unset PHI_OMP_NUM_THREADS
unset PHI_KMP_AFFINITY
unset I_MPI_MIC
unset I_MPI_FALLBACK
unset LD_LIBRARY_PATH
