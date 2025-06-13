#
# *******************************************************************
#
#	C-DAC Tech Workshop : hyPACK-2013
#                October 15-18, 2013
#
#   Created             : August-2013
#
#   E-mail              : hpcfte@cdac.in     
#
# *******************************************************************

##Uer configurable options #####

F77          =  mpif77

LIBS        =  -lm

### End User configurable options ###

default: run

#For DAY-2 spmd examples only......

#OBJECTS = allgather.o
#OBJECTS = allreduce.o
#OBJECTS = broadcast.o
#OBJECTS = gatherv.o
#OBJECTS = pie-collective.o
#OBJECTS = reduce.o
OBJECTS = scatter.o


run: $(OBJECTS)
	$(F77) -o run $(OBJECTS) $(LIBS)

.f.o:
	$(F77) -c $<


clean:
	\rm -f *.o run

