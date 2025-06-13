#
# *******************************************************************
#
#	C-DAC Tech Workshop : hyPACK-2013
#                 October 15-18, 2013
#
# *******************************************************************
#   Created             : Augusr 2013
#
# *******************************************************************
#
##### User configurable options #####

MPIR_HOME  = 
#F77        = mpxlf
F77        = mpif77

OPTFLAGS   = 

MPILIB     =
LIBS       = 

### End User configurable options ###

default: run

#For DAY-2 examples only 

OBJECTS=vv_mult_blk_cyclic.o
#OBJECTS=vv_mult_blkstp_nonunf.o
#OBJECTS=vv_mult_blkstp_unf.o
#OBJECTS=mat_infnorm_blkstrip.o
#OBJECTS=mv_mult_blkstrip.o
#OBJECTS=mv_mult_checkerboard.o
#OBJECTS=mm_mult_cartesian.o


run: $(OBJECTS)
	$(F77) -o run $(OBJECTS) $(LIBS) $(MPILIB) 

.f.o:
	$(F77) -c $<

	
clean: 
	\rm -f *.o run

