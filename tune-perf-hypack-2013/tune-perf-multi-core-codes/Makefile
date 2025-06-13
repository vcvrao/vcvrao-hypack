#
#
# *******************************************************************
#	            C-DAC Tech Workshop : hyPACK-2013
#                             October 15-18, 2013
#
#
# *******************************************************************
#  
#   Created           August 2013 
#
# *******************************************************************

#
#--------------------------------------------------------------------------
#                      C and Fortran Compilers and Linkers
#--------------------------------------------------------------------------

CC            =  icc
CCLINKER      =  $(CC)

F77           =  ifort
#Uncomment the following line to use BLAS routines from IBM ESSLSMP libraies
#F77           =  xlf_r
F77LINKER     =  $(F77)


#-------------------------------------------------------------------------
#                Flags used for compilation and linking
#-------------------------------------------------------------------------

CFLAGS        = 
FFLAGS        =  
#Uncomment the following line to use BLAS routines from IBM ESSLSMP libraies
#FFLAGS        =  -qextname -qomp

LINKFLAGS     = 
#Uncomment the following line to use BLAS routines from IBM ESSLSMP libraies
#LINKFLAGS     =  -qomp -qstrict -bmaxdata:0x80000000

#Uncomment the following line if compiler optimizations are required
#OPTFLAGS      =  -qstrict -O3

#Uncomment one of the following based on the program name for compilation
FOBJECTS=single-core-prg-loop-interchange-center.o
#FOBJECTS= single-core-prg-loop-unroll-interchange-block.o  
#FOBJECTS=single-core-prg-tlb-miss.o
#FOBJECTS=single-core-prg-loop-interchange.o 
#FOBJECTS=single-core-prg-memory-manage.o
#FOBJECTS=single-core-prg-tryparammod.o
#FOBJECTS=single-core-prg-if-goto.o
#FOBJECTS=single-core-prg-tryparamorig.o

COBJECTS=single-core-prg-loop-unroll.o
#COBJECTS=single-core-prg-neighbour-data-dependency.o

#-------------------------------------------------------------------------
#                      Libraries and Executables
#-------------------------------------------------------------------------

#Uncomment the following line to use BLAS routines from IBM ESSL libraries
#BLASLIBS      =  -lessl
#Uncomment the following line to use BLAS routines from IBM ESSLSMP libraies
#BLASLIBS      =  -lesslsmp

LIBS          =  $(BLASLIBS) -lm
EXECS         =  runc runf

default: $(EXECS)

runc:$(COBJECTS)
	$(CCLINKER) $(OPTFLAGS) $(LINKFLAGS) -o runc $(COBJECTS) $(LIBS)

runf:$(FOBJECTS)
	$(F77LINKER) $(OPTFLAGS) $(LINKFLAGS) -o runf $(FOBJECTS) $(LIBS)

.c.o:
	$(CC) -c $(OPTFLAGS) $(CCFLAGS) $<

.f.o:
	$(F77) -c $(OPTFLAGS) $(FFLAGS) $<

clean:
	\rm -rf *.o core runc runf
