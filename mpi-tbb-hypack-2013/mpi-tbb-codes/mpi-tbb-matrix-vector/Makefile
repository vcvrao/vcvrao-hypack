#

#   C-DAC Tech Workshop : hyPACK-2013
#         October 15-18, 2013
#
# Makefile 
# use make all to compile and link code of MPI + intel TBB.
#
#

## TBB LIB and INCLUDE
TBB_LIB=/opt/intel/tbb/2.0/lib 
TBB_LIB_FLAGS=-ltbb
TBB_INC=/opt/intel/tbb/2.0/include

## Compiler MPI C++ compiler
CC = mpicxx

#Optimization flags
#[to remove warning caused by conflict of mpi.h with stdio.h]
OPT_FLAGS=-DMPICH_IGNORE_CXX_SEEK 

OBJECTS= mpi_tbb_matrix_vector_multiply

all:$(OBJECTS)

$(OBJECTS): $(OBJECTS).o
	$(CC) -o $(OBJECTS) ./$(OBJECTS).o -L$(TBB_LIB) $(TBB_LIB_FLAGS) $(OPT_FLAGS)

$(OBJECTS).o: $(OBJECTS).cpp
	$(CC) -I$(TBB_INC) -c $(OBJECTS).cpp $(OPT_FLAGS)

## clean bin and object files
clean:
	$(RM) -f ./$(OBJECTS)  
	$(RM) -f  *.o   
