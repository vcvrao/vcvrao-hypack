#
#
#	C-DAC Tech Workshop : hyPACK-2013
#              October 15-18, 2013
#
##### User configurable options #####

MPIR_HOME  = 
F77        = mpxlf

OPTFLAGS   = 

MPILIB     =
LIBS       = 

### End User configurable options ###

default: run

OBJECTS= sp_input.o sp_main.o sp_outg.o sp_row.o sp_scat.o sp_sermv.o

run:clean $(OBJECTS)
	$(F77) -o run $(OBJECTS) $(LIBS) $(MPILIB) 

.f.o:
	$(F77) -c $<

	
clean: 
	\rm -f *.o run

