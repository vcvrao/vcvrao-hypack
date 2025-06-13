#
# ***************************************************************
#
#		C-DAC Tech Workshop : hyPACK-2013
#                           October 15-18, 2013
#
#   Created             : August 2013
#
#   E-mail              : hpcfte@cdac.in     
#
# ***************************************************************
#
# Makefile 
# use make all to compile and link code of TBB.
#
#**********************************************************

## TBB LIB and INC
TBB_LIB=/opt/intel/tbb/2.0/lib 
TBB_LIB_FLAGS=-ltbb
TBB_INC=/opt/intel/tbb/2.0/include

## Compilers 
CC=g++
#CC=icc

#Optimization flags
OPT_FLAGS=-O3

#OBJECTS= VectorDotOp
#OBJECTS= tbb-matmat-multiply
#OBJECTS= tbb-matrix-vector
OBJECTS= tbb-vectorvector
#OBJECTS= ArrayMult 
#OBJECTS= ParrelReduceMinMax


all:$(OBJECTS)

$(OBJECTS): $(OBJECTS).o
	$(CC) -o $(OBJECTS) ./$(OBJECTS).o -L$(TBB_LIB) $(TBB_LIB_FLAGS)

$(OBJECTS).o: $(OBJECTS).cpp
	$(CC) -I$(TBB_INC) -c $(OPT_FLAGS ) $(OBJECTS).cpp

## clean bin and object files
clean:
	$(RM) -f ./$(OBJECTS)  
	$(RM) -f  *.o  
