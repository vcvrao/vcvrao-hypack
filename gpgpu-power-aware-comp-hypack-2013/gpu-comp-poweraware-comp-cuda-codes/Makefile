#............ HeGaPa-2012 ...........................................
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

#OBJECTS = cuda-hello-world.cu
#OBJECTS = cuda-device-query.cu
#OBJECTS = cuda-find-device.cu
#OBJECTS = cuda-simple-kernel-parameter-prog.cu
#OBJECTS = cuda-vector-vector-addition-blocks.cu
#OBJECTS = cuda-vector-vector-addition-grid-blocks-threads.cu

#----------------------------------------------------------------------

run:$(OBJECTS)
	$(CCLINKER) $(CCFLAGS) $(OPTFLAGS) -o $@ $(OBJECTS) $(LIBS) 


clean:
	\rm -rf *.o run

