#
#**********************************************************************
#
#
#               C-DAC Tech Workshop : hyPACK-2013
#                           October 15-18, 2013
#
# Created               : August-2013
# 
#       
#************************************************************************

# set environment variables
#export LD_LIBRARY_PATH=/opt/intel/lib/mic/:/opt/intel/mkl/lib/mic/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/intel/mic/lib64/:/opt/intel/lib/mic:/opt/intel/mkl/lib/mic/:${LD_LIBRARY_PATH}
#export MKL_NUM_THREADS=236
#export KMP_AFFINITY=granularity=fine,compact
# compilation of source code
#make -f Makefile.NATIVE clean
#make -f Makefile.NATIVE
#execution of source code
#./run 16000
./run 1000000000 236
#unset env variables
 unset LD_LIBRARY_PATH
#unset MKL_NUM_THREADS
#unset KMP_AFFINITY


