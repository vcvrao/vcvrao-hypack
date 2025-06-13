#
# *******************************************************************
#		 C-DAC Tech Workshop HyPack-2013 
#		     July 15-18,2013
#
# *******************************************************************
#   
#   Created             : Aug 2013
#                        
#
# *******************************************************************

#
ALL: default

###  User configurable options ###
CC          = mpicc
CLINKER     = $(CC)


MPILIB     = 
LIBS       = -lm -lpthread

CFLAGS    = 


### End User configurable options ###

EXECS	  =  run 

OBJECTS=mpi-pthreads-helloworld.o
#OBJECTS=mpi-pthreads-infinity-norm.o
#OBJECTS=mpi-pthreads-matrix-vector.o
#OBJECTS=mpi-pthreads-pie-collective.o

default: $(EXECS)

run : $(OBJECTS) 
	$(CLINKER) $(CFLAGS) -o run  $(OBJECTS) $(LIBS) $(MPILIB)


.c.o:
	$(CC) -c $*.c


clean : 
	$(RM) *.o core run 
