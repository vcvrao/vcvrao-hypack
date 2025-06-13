#	C-DAC Tech Workshop : hyPACK-2013
#           October 15-18, 2013
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


i=1
while test "$i" -lt 6 ;do
  mpiexec.hydra -np 2 -machinefile mpi_hosts_xeon_phi ./run 1024 $i 10
 echo "$i"
let i++
#i=$((i + 1))
done

