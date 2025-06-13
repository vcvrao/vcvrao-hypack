SHELL=/bin/sh
#---------------------------------------------------------------------------
# This is the C compiler used for C programs
#---------------------------------------------------------------------------

CC = g++



#---------------------------------------------------------------------------
#  Set up location of OpenCL sdk library and runtime installed location
#---------------------------------------------------------------------------
CLHOME = /usr/local/cuda
CLLIB  = -L$(CLHOME)/lib64
CLINC = -I$(CLHOME)/include/
CLLINKFLAGS=-lOpenCL -fopenmp -lm


#---------------------------------------------------------------------------
# Destination of executables, relative to subdirs of the main directory. .
#---------------------------------------------------------------------------



# Name of set of example program
OBJECTS =  Mat_Mat_Mult_Openmp_Opencl_GlobalMemory   Mat_Mat_Mult_Openmp_Opencl_SharedMemory  Vect_Vect_Add_Openmp_Opencl

all: $(OBJECTS)

$(OBJECTS):%:%.c 
	$(CC) $(CLLINKFLAGS) $(CLLIB) $(CLINC) -o $@ $@.c  
clean : 
	rm -rf *.o $(OBJECTS)

