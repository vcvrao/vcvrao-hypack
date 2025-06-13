#
#******************************************************************
#
#	C-DAC Tech Workshop : hyPACK-2013
#               October 15-18, 2013
#
#  Created         : August 2013 
#
#*****************************************************************
#
#--------------------------------------------------------------------------
#                      C and Fortran Compilers and Linkers
#--------------------------------------------------------------------------

F77           =  ifort
F77LINKER     =  $(F77)


#FOBJECTS=mathlib-core-horner-rule.f
FOBJECTS=mathlib-core-mat-mat-mult-dotproduct.f
#FOBJECTS=mathlib-core-linear-system-gauss-solver.f
#FOBJECTS=mathlib-core-mat-vect-mult-columnwise.f
#FOBJECTS=mathlib-core-mat-mat-mult-daxpy.f
#FOBJECTS=mathlib-core-mat-vect-mult-rowwise.f


#-------------------------------------------------------------------------
#                Flags used for compilation and linking
#-------------------------------------------------------------------------

CFLAGS        = 
FFLAGS        =  

LINKFLAGS     =  

#Uncomment one of the following based on the program name for compilation

#-------------------------------------------------------------------------
#                      Libraries and Executables
#-------------------------------------------------------------------------
LIBS          =  -lm
EXECS         =  runf 

runf:$(FOBJECTS)
	$(F77LINKER) $(OPTFLAGS) $(LINKFLAGS) -o runf $(FOBJECTS) $(LIBS)

.f.o:
	$(F77) -c $(OPTFLAGS) $(FFLAGS) $<

clean:
	\rm -rf *.o core runc runf
