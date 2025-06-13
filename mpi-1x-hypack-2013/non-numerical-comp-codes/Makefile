#
# *******************************************************************
#
#	C-DAC Tech Workshop : hyPACK-2013
#                October 15-18, 2013
#
# *******************************************************************
#   Created             : August 2013 
#
# *******************************************************************

##### User configurable options #####

MPIR_HOME  = 

#Comment the following line for compiling sum_rma.f
CC        = mpicc

#Uncomment the following line for compiling sum_rma.f

OPTFLAGS   = 

MPILIB     =
LIBS       = 

### End User configurable options ###

default: run

#For DAY-3 examples only 

OBJECTS=mpic-samplesort.o


run:$(OBJECTS)
	$(CC) -o run $(OBJECTS) $(LIBS) $(MPILIB) 
	
.f.o:
	$(CC) -c $<

	
clean: 
	\rm -f *.o run

