BIN=./../../bin

########################
# Compliler & Loaders 
#######################

CC = icc
CC_GPU = nvcc
CCFLAGS = -O3

##############################
# Path to Intel MKL libraries
##############################

CBLAS_FLAGS= -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lpthread -lm
#CBLAS_INCLUDE=  /opt/intel/Compiler/11.1/069/mkl/include
CBLAS_INCLUDE=  /opt/ohpc/pub/syssoftware/intel/compilers_and_libraries_2017.4.196/linux/mkl
#CBLAS_LIB=  /opt/intel/Compiler/11.1/069/mkl/lib/em64t/  /opt/intel/Compiler/11.1/069/lib/intel64/libiomp5.so
CBLAS_LIB=  /opt/ohpc/pub/syssoftware/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64  /opt/ohpc/pub/syssoftware/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64/libiomp5.so

####################
# Cuda Libraries
####################

CUDA_INCLUDE= /usr/local/cuda/include/
CUDA_LIBS= /usr/local/cuda/lib64/  #$(LIBS)

