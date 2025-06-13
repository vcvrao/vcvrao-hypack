#
# *******************************************************************
#
#	C-DAC Tech Workshop : hyPACK-2013
#                  October 15-18, 2013
#
# *******************************************************************
#  Created     :August 2013 
#
# *******************************************************************

#---------------------------------------------------------------------
#		Machines Architecture :
#---------------------------------------------------------------------
FCOMPILER   =  mpif90
FLINKER     =  $(FCOMPILER) 
COPTION     = 
OPTFLAGS    =  

EXECS       =  run

default : $(EXECS)

#---------------------------------------------------------------------
#		Object files ...
#---------------------------------------------------------------------
#


#OBJECTS=f90-hello-world.f90
#OBJECTS=f90-leastsquare.f90
#OBJECTS=f90-linear-eqn.f90
#OBJECTS=f90-matrixvector-mult.f90
OBJECTS=f90-mpi-io-single-file.f90
#OBJECTS=f90-newton-raphson-root.f90
#OBJECTS=f90-root-quadratic-eqn.f90
#OBJECTS=f90-welcome.f90
#OBJECTS=Gauss_Modules.f90
#OBJECTS=f90-gauss-elimination.f90

#----------------------------------------------------------------------

run:$(OBJECTS) 
	$(FLINKER) $(OPTFLAGS) -o $@ $(OBJECTS)

.f90.o:
	$(FCOMPILER) -c $(OPTFLAGS) $(FFLAGS) $<

clean:
	\rm -rf *.o *.mod *.M $(EXECS) testfile

