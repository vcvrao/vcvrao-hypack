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
#
#Makefile for running cuda practice programs
#---------------------------------------------------------------------
#               Machines Architecture :
#---------------------------------------------------------------------
CC   =  nvcc
CCLINKER   =  $(CC)
COPTION     =
OPTFLAGS    =  
CCFLAGS     =  -arch sm_20
#CCFLAGS     = -deviceemu
LIBS        =
EXECS       =  run
#---------------------------------------------------------------------
#               Object files ...
#---------------------------------------------------------------------

OBJECTS = CudaPoissonEquation.cu
#OBJECTS = cuda-vector-vector-addition_SharedMemory.cu
#OBJECTS = cuda-vector-vector-multiplication_GlobalMemory.cu
#OBJECTS = cuda-vector-vector-multiplication_SharedMemory.cu 
#OBJECTS = cuda-matrix-vector-multiplication.cu
#OBJECTS = cuda-PI-computation.cu
#OBJECTS =  cuda-prefix-sum.cu  
#OBJECTS = cuda-transpose-matrix.cu
#OBJECTS = cuda-infinity-norm.cu 
#OBJECTS=cuda-matrix-matrix-multiplication.cu
#OBJECTS=CudaConjugateGradient.cu
#OBJECTS=cuda-jacobi.cu
#OBJECTS=CUBlasJacobi.cu

#OBJECTS = cuda-vector-vector-multiplication-mGPU.cu
#OBJECTS=cuda-matrix-matrix-multiplication-mGPU.cu


#----------------------------------------------------------------------

run:$(OBJECTS)
	$(CCLINKER) $(CCFLAGS) $(OPTFLAGS) -o $@ $(OBJECTS) $(LIBS) 


clean:
	\rm -rf *.o run

