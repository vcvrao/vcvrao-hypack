#
# *******************************************************************
#
#	C-DAC Tech Workshop : hyPACK-2013
#               October 15-18, 2013
#
# *******************************************************************
#   Created             : August 2013 
#
# *******************************************************************
#
##### User configurable options #####

CC          =  mpicxx

LIBS        =  -lm 

CFLAGS      = -DMPICH_IGNORE_CXX_SEEK


### End User configurable options ###

default: run


OBJECTS=mpicpp-hello-world.C
#OBJECTS=mpicpp-pie-pt-to-pt.C
#OBJECTS=mpicpp-sum-associative-fanin-blocking-tree.C
#OBJECTS=mpicpp-sum-pt-to-pt.C
#OBJECTS=mpicpp-sum-ring-topology.C
#OBJECTS=sum-pt-to-pt.C

run: $(OBJECTS)
	$(CC) -o run $(CFLAGS) $(OBJECTS) $(LIBS)  

.c.o:
	$(CC) -c $<

	
clean: 
	\rm -f *.o run

