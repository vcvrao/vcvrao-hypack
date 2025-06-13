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
#Makefile for running cuda bandwidth assignment
#---------------------------------------------------------------------
#               Machines Architecture :
#---------------------------------------------------------------------
CC   = /usr/local/cuda/bin/nvcc
CCLINKER   =  $(CC)
COPTION     =
OPTFLAGS    =
CCFLAGS     =
#CCFLAGS     = -deviceemu
LIBS        =
EXECS       = bandwidthExe 
#---------------------------------------------------------------------
#               Object files ...
#---------------------------------------------------------------------

OBJECTS = CudaBandwidthCal-v1.1.cu 
#----------------------------------------------------------------------

$(EXECS):$(OBJECTS)
	$(CCLINKER) $(CCFLAGS) $(OPTFLAGS) -o $@ $(OBJECTS) $(LIBS) 


clean:
	\rm -rf *.o $(EXECS) 


