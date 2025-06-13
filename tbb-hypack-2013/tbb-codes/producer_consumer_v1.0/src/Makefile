#---------------------------------------------------------------------------#
#		C-DAC Tech Workshop : hyPACK-2013                           #
#                           October 15-18, 2013                              #
#                                                                           #
#                        producer_consumer_v1.0                             #
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


TBB_LIB=/opt/intel/tbb/2.0/lib
TBB_LIB_FLAGS=-ltbb
TBB_INC=/opt/intel/tbb/2.0/include/


## Compilers
CC=g++
OPT_FLAGS=-O3
all:prodcon

main-producer-consumer-v.1.0.o : main-producer-consumer-v.1.0.cc ../include/Consumer.h ../include/Producer.h ../include/Proto.h ../include/headerfiles.h
	$(CC) -c  main-producer-consumer-v.1.0.cc  -I$(TBB_INC)

prod_cons_functions.o : ../lib/prod_cons_functions.cc ../include/Consumer.h ../include/Producer.h ../include/Proto.h ../include/headerfiles.h
	$(CC) -c ../lib/prod_cons_functions.cc -I$(TBB_INC)

prodcon : prod_cons_functions.o main-producer-consumer-v.1.0.o
	$(CC) -g  prod_cons_functions.o main-producer-consumer-v.1.0.o  -o prodcon -L$(TBB_LIB) $(TBB_LIB_FLAGS)

clean:
	$(RM) -f ./prodcon
	$(RM) -f  *.o

