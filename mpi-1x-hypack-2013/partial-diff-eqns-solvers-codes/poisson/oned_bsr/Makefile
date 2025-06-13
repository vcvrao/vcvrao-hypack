ALL: default
##### User configurable options #####

SHELL       = /bin/sh
F77          = mpif77
FLINKER     = $(F77)
OPTFLAGS    =   

### End User configurable options ###

CFLAGS	  = $(OPTFLAGS) 
CFLAGSMPE = $(CFLAGS) -I$(MPE_DIR)
CCFLAGS	  = $(CFLAGS)
FFLAGS	  = $(OPTFLAGS)
EXECS	  =  run 

default: $(EXECS)

TARGETS : run 
all     : $(TARGETS)

OBJECTS = 1dbsr.o

run: $(OBJECTS)
	$(FLINKER) $(OPTFLAGS) -o run $(OBJECTS)

.f.o:
	$(F77) $(FFLAGS) -c $*.f

clean : 
	\rm -rf *.o core run
