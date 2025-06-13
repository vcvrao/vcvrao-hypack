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

OBJECTS  = cuda-pageble-memory.cu 
#OBJECTS = cuda-pinned-memory.cu
#OBJECTS = single-cuda-stream.cu
#OBJECTS = multiple-cuda-streams.cu
#OBJECTS = multipleKernels-multiGPU-streams-matrix-matrix-comp.cu
#OBJECTS = multipleKernels-singleGPU-matrix-vector-multiplication.cu
#OBJECTS = mksg-sm-cuda-stream-Matrix-Matrix-Addition.cu

#----------------------------------------------------------------------

run:$(OBJECTS)
	$(CCLINKER) $(CCFLAGS) $(OPTFLAGS) -o $@ $(OBJECTS) $(LIBS) 


clean:
	\rm -rf *.o run

