#---------------------------------------------------------------------------#
#                       tbb-producer-consumer_v1.1                          #
#                            Makefile                                       #
#---------------------------------------------------------------------------#
# Description : This file contains the paths, targets and dependencies to   #
#               be used while compilation                                   #
#               Use "make" to compile and link                              #
#               and Use "make clean" to clean                               #
#                                                                           #
# Created     : August,2013                                                 #
#                                                                           #
# E-mail      : hpcfte@cdac.in                                              #
#---------------------------------------------------------------------------#


TBB_LIB=/opt/intel/tbb/2.0/lib/libtbb.so /opt/intel/tbb/2.0/lib/libtbbmalloc.so
##TBB_LIB_FLAGS=-ltbb -ltbbmalloc
TBB_INC=/opt/intel/tbb/2.0/include/

## Compilers
CC=g++
OPT_FLAGS=-O3
all:run

main.o : main.cc ../include/Consumer.h ../include/Producer.h ../include/Proto.h ../include/headerfiles.h ../include/fun_declaration.h
	$(CC) -c  main.cc -I$(TBB_INC)

prod_cons_functions.o : ../lib/prod_cons_functions.cc ../include/Consumer.h ../include/Producer.h ../include/Proto.h ../include/headerfiles.h ../include/fun_declaration.h
	$(CC) -c ../lib/prod_cons_functions.cc -I$(TBB_INC)

run: prod_cons_functions.o  main.o
	$(CC) -g  prod_cons_functions.o  main.o  -o run $(TBB_LIB) 

clean:
	$(RM) -f ./run
	$(RM) -f  *.o

