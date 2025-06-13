
#---------------------------------------------------------------------------#
#		C-DAC Tech Workshop : hyPACK-2013                           #
#                           October 15-18, 2013                             #
#                                                                           #
#                        tbb_producer_consumer_v1.0                         #
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

#!/bin/sh

include ./include_header/header.h

EXE = tbbwork
OBJS = tbb-producer-consumer-CQ.o
SOURCE = ./src/tbb-producer-consumer-CQ.cc

#CC  - Cpp compiler
CC = g++
#CC = icc

# CPPFLAGS - Cpp compilation flags
CPP_FLAGS = -O3 -Wall

# TBB_INCLUDES     - any -I arguments required for compiling
TBB_INCLUDES = -I/opt/intel/tbb/2.0/include

# TBB_LIBS=     - any -L and -l arguments required for linking
TBB_LIBS = /opt/intel/tbb/2.0/lib/libtbb.so

tbbwork:$(OBJS)
	$(CC) $(CPP_FLAGS) -o $(EXE) $(OBJS) $(TBB_LIBS) 

tbb-producer-consumer-CQ.o: $(SOURCE) 
	$(CC) -c $(SOURCE) $(TBB_INCLUDES) 

clean:
	rm -f *.o $(EXE)
