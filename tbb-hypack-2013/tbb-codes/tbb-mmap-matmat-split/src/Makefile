
#---------------------------------------------------------------------------#
#		C-DAC Tech Workshop : hyPACK-2013                           #
#                           October 15-18, 2013                             #
#                                                                           #
#                        tbb-mmap-matmat-split.cc                           #
#                            Makefile                                       #
#---------------------------------------------------------------------------#
# Description : This file contains the paths, targets and dependencies to   #
#               be used while compilation                                   #
#               Use "make" to compile and link                              #
#               and Use "make clean" to clean                               #
#                                                                           #
#                                                                           #
# Created     : August-2013                                                 #
#                                                                           #
# E-mail      : hpcfte@cdac.in                                              #
#                                                                           #
#---------------------------------------------------------------------------#

# Makefile for  compile TBB code
# use make all to compile and link code .
#

##TBB  LIB and INC variable  path set according to 
## your system where TBB is install 

TBB_LIB=/opt/intel/tbb/2.0/lib 
TBB_LIB_FLAGS=-ltbb
TBB_INC=/opt/intel/tbb/2.0/include


## Compilers 
CC=g++
#CC=icc
OPT_FLAGS=-O3
all: mymat

print.o : print.cc ../include/define.h ../include/sysheader.h ../include/proto.h 
	$(CC) -c print.cc -I$(TBB_INC)

input.o : input.cc ../include/define.h ../include/sysheader.h ../include/proto.h
	$(CC) -c input.cc -I$(TBB_INC)

memory.o : memory.cc ../include/define.h ../include/sysheader.h ../include/proto.h
	$(CC) -c memory.cc -I$(TBB_INC)

multiply.o : multiply.cc ../include/define.h ../include/sysheader.h ../include/proto.h
	$(CC) -c multiply.cc -I$(TBB_INC)

main.o : main.cc ../include/define.h ../include/sysheader.h ../include/proto.h
	$(CC) -c main.cc -I$(TBB_INC)

mymat : print.o input.o memory.o multiply.o main.o
	$(CC) -g  print.o input.o memory.o multiply.o main.o -o mymat -L$(TBB_LIB) $(TBB_LIB_FLAGS)





#all:$(OBJECTS)
#TBB_LIB=/opt/intel/tbb/2.0/lib 
#TBB_LIB_FLAGS=-ltbb
#TBB_INC=/opt/intel/tbb/2.0/include

#$(OBJECTS): $(OBJECTS).o
#	$(CC) -o $(OBJECTS) ./$(OBJECTS).o -L$(TBB_LIB) $(TBB_LIB_FLAGS)

#$(OBJECTS).o: $(OBJECTS).cc
#	$(CC) -I$(TBB_INC) -c $(OPT_FLAGS )$(OBJECTS).cc

## clean bin and object files
clean:
#	$(RM) -f ./$(OBJECTS)  
	$(RM) -f ./mymat  
	$(RM) -f  *.o  
