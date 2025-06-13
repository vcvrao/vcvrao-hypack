#
#  C-DAC Tech Workshop : hyPACK-2013
#         October 15-18, 2013
#
#  Created       :  August-2013
#
#  E-mail       :  hpcfte@cdac.in     
#
#!/bin/bash

export OMP_NUM_THREADS = 61 
i=1
while test "$i" -lt 62 ;do
  mpiexec.hydra -np 2 -machinefile mpi_hosts ./run 1024 $i 10
 echo "$i"
let i++
#i=$((i + 1))
done

