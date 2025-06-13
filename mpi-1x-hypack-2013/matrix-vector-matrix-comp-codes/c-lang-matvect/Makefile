#
# *******************************************************************
#
#		C-DAC Tech Workshop : hyPACK-2013
#                           October 15-18, 2013
#
# *******************************************************************
#
#   Created             : August 2013 
#
# *******************************************************************

##### User configurable options #####

#CC          =  mpcc
CC          =  mpicc

LIBS        =  -lm 

### End User configurable options ###

default: run

#For DAY-2 spmd examples only......

#OBJECTS = vv_mult_blkstp_unf.o
#OBJECTS = vv_mult_blkstp_nonunf.o
#OBJECTS = vv_mult_blk_cyclic.o
#OBJECTS = mat_infnorm_blkstrip.o
#OBJECTS = mv_mult_mult_blkstrip.o
#OBJECTS = mv_mult_mult_checkerboard.o
#OBJECTS = mm_mult_cartesian.o
#OBJECTS = mm_mult_cannon.o
OBJECTS = mm_mult_fox.o

run: $(OBJECTS)
	$(CC) -o run $(OBJECTS) $(LIBS)  

.c.o:
	$(CC) -c $<

	
clean: 
	\rm -f *.o run

