export LD_LIBRARY_PATH=/opt/intel/lib/mic/:/opt/intel/mkl/lib/mic/:${LD_LIBRARY_PATH}
#source /opt/intel/bin/compilervars.sh intel64
export MIC_ENV_PREFIX=PHI
#export PHI_OMP_NUM_THREADS=240
export PHI_KMP_AFFINITY=granularity=fine,compact

