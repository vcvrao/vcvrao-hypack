#
#   C-DAC Tech Workshop : hyPACK-2013
#          October 15-18, 2013
#
#   Created           :  August-2013
#
#   E-mail            :  hpcfte@cdac.in     
#
#!/bin/bash

# set environment variables 
export LD_LIBRARY_PATH=/opt/intel/lib/mic/:/opt/intel/mkl/lib/mic/:${LD_LIBRARY_PATH}
export OMP_NUM_THREADS=236
export KMP_AFFINITY=granularity=fine,compact

#execution of source code
#./run 16000
./vector-vector-addition-scale-openmp-native 236 

#unset env variables
unset LD_LIBRARY_PATH
unset OMP_NUM_THREADS
unset KMP_AFFINITY=granularity
