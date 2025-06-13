export OFFLOAD_REPORT=2
export MIC_ENV_PREFIX=MIC
#export MIC_OMP_NUM_THREADS=240
export MIC_KMP_AFFINITY=granularity=fine,compact
#export MIC_KMP_AFFINITY=granularity=fine,balanced
#export MIC_KMP_AFFINITY=granularity=fine,scatter

make -f Makefile_VEC_VEC_ADD_AOS_SOA clean  
make -f Makefile_VEC_VEC_ADD_AOS_SOA 
make -f Makefile_MAT_MAT_ADD_SOA2D 

