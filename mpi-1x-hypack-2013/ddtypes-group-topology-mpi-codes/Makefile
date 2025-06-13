#
# *****************************************************************
#
#		C-DAC Tech Workshop : hyPACK-2013
#                           October 15-18, 2013
#
# *******************************************************************
#   Created             : August 2013 
#
# *******************************************************************

ALL: default
##### User configurable options #####

SHELL     = /bin/sh
ARCH      = 
COMM      = 

CC          = mpicc
OPTFLAGS    = 

### End User configurable options ###

EXECS      =  run

default: $(EXECS)

all  : run

#For DAY-1 examples only......

#OBJECTS = array-contiguous-memory.o
#OBJECTS = multiple-communicators-cartesian.o
#OBJECTS = upper-traingular-portion-ddtype.o
#OBJECTS = array-non-contiguous-memory.o
#OBJECTS = memory-pack-unpack.o 
#OBJECTS = multiple-communicators-split.o
#OBJECTS = general-derived-datatype.o 
#OBJECTS = multiple-communicators.o
OBJECTS = split1.o

run: $(OBJECTS)
	$(CC) -o run $(OBJECTS) -lm 

.c.o:
	$(CC) -c $<

clean: 
	\rm -f *.o run

