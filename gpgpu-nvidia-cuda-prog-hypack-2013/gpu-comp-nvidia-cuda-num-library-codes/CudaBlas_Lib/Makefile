#
#
# *******************************************************************
#
#		C-DAC Tech Workshop : hyPACK-2013
#                           October 15-18, 2013
#
# *******************************************************************
#   Created             : August 2013
#
#   E-mail              : hpcfte@cdac.in     
#
# Makefile for  level1 BLAS (CUBLAS) and CUBLAS functions
# use make all to compile and link code of CPU & GPU.
#
#
## GPU
GPU_CC=nvcc
GPU_LD=nvcc
CUDA_DIR = /usr/local/cuda
CUDALIB = -L$(CUDA_DIR)/lib -lcublas
CUDABLAS_INC  = -I$(CUDA_DIR)/include 

all = cublas1 cublas2 cublas3

##level1
cublas1: cuda_blas1.o test_cublas1.o
	$(GPU_CC) -o cublas1 cuda_blas1.o test_cublas1.o $(CUDALIB) 

test_cublas1.o: test_cublas1.c
	$(GPU_CC) $(CUDABLAS_INC) -c test_cublas1.c 

cuda_blas1.o: cuda_blas1.c
	$(GPU_CC) $(CUDABLAS_INC) -c cuda_blas1.c 
	

##level2	
cublas2: cuda_blas2.o test_cublas2.o
	$(GPU_CC) -o cublas2 cuda_blas2.o test_cublas2.o $(CUDALIB) 

test_cublas2.o: test_cublas2.c
	$(GPU_CC) $(CUDABLAS_INC) -c test_cublas2.c 

cuda_blas2.o: cuda_blas2.c
	$(GPU_CC) $(CUDABLAS_INC) -c cuda_blas2.c 
	
##level3	
cublas3: cuda_blas3.o test_cublas3.o
	$(GPU_CC) -o cublas3 cuda_blas3.o test_cublas3.o $(CUDALIB) 

test_cublas3.o: test_cublas3.c
	$(GPU_CC) $(CUDABLAS_INC) -c test_cublas3.c 

cuda_blas3.o: cuda_blas3.c
	$(GPU_CC) $(CUDABLAS_INC) -c cuda_blas3.c 	



## clean bin and object files
clean:
	\rm -rf cublas1 cublas2 cublas3
	 \rm -rf *.o
