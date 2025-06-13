#
# ***************************************************************
#
#	C-DAC Tech Workshop : hyPACK-2013
#              October 15-18, 2013
#
#   Created             : August 2013
#
# ****************************************************************
#
# User configurable options #####

#CC         =  mpcc

CC          =  mpicc

LIBS        =  -lm

### End User configurable options ###

default: run

#For DAY-2 spmd examples only......

#OBJECTS = allgather.o
#OBJECTS = allreduce.o
#OBJECTS = alltoall.o
OBJECTS = broadcast.o
#OBJECTS = gatherv.o
#OBJECTS = pie-collective.o
#OBJECTS = reduce.o
OBJECTS = scatter.o

run: $(OBJECTS)
	$(CC) -o run $(OBJECTS) $(LIBS)

.c.o:
	$(CC) -c $<


clean:
	\rm -f *.o run

