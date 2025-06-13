#create a directory data
mkdir -p data

#set environment variables
 export MIC_ENABLE_PREFIX=MIC
 export MIC_OMP_NUM_THREADS=236
 export MIC_KMP_AFFINITY=granularity=fine,compact


#compile the codes
 make -f Makefile_mmap.OFFLOAD clean
 make -f Makefile_mmap.OFFLOAD

#execute

 ./run 4096

make -f Makefile_mmap.OFFLOAD clean
