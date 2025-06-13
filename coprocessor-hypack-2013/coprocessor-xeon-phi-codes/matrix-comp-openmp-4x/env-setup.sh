#                  CDAC Tech Workshop - hyPACK 2013
#			Oct 15 - 18 , 2013
source /opt/intel/bin/compilervars.sh intel64
export OFFLOAD_REPORT=2
export MIC_ENV_PREFIX=MIC
#export MIC_OMP_NUM_THREADS=240
export MIC_KMP_AFFINITY=granularity=fine,compact
#export MIC_KMP_AFFINITY=granularity=fine,balanced
#export MIC_KMP_AFFINITY=granularity=fine,scatter
export H_TRACE=1
export MIC_LD_LIBRARY_PATH=/opt/intel/lib/mic/:/opt/intel/mkl/lib/mic/:${MIC_LD_LIBRARY_PATH}
#compilations

make -f Makefile_openmp4x clean 
make -f Makefile_openmp4x 


#execution

./run 


