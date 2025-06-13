#
# *******************************************************************
#
#		C-DAC Tech Workshop : hyPACK-2013
#                           October 15-18, 2013
#
# *******************************************************************
#   Created             : August 2013 
#
# *******************************************************************

##### User configurable options #####

CC          =  mpicc

LIBS        =  -lm 

### End User configurable options ###

default: run


OBJECTS=sum_associative_fanin_nonblocking_tree.c
#OBJECTS=sum_ring_topology-non-block.c

run: $(OBJECTS)
	$(CC) -o run $(OBJECTS) $(LIBS)  

.c.o:
	$(CC) -c $<

	
clean: 
	\rm -f *.o run

