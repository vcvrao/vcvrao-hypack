
#set environment variables
export OMP_NUM_THREADS=236
#export KMP_AFFINITY=granularity=fine,scatter
#export KMP_AFFINITY=granularity=fine,balanced
export KMP_AFFINITY=granularity=fine,compact


#execute
./run 8192 236 3

unset OMP_NUM_THREADS
unset KMP_AFFINITY
