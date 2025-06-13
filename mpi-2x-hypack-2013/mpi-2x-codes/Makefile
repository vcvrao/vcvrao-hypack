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
# *******************************************************************

ALL: default
##### User configurable options #####

CC          = mpicc

### End User configurable options ###

EXECS      =  run

default: $(EXECS)

all  : run


#OBJECTS=mpi-io-multiple-files.c
#OBJECTS=remote-memory-access-pie-comp.c
OBJECTS=remote-memory-access-sum.c

run: $(OBJECTS)
	$(CC) -o run $(OBJECTS)  

.c.o:
	$(CC) -c $<

	
clean: 
	\rm -f *.o run  ParallelIO*

