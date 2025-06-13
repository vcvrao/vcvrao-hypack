#---------------------------------------------------------------------------#
#		C-DAC Tech Workshop : hyPACK-2013                           #
#                           October 15-18, 2013                             #
#                                                                           #
#                        tbb-vec-vec-malloc                                 #
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

TBB_LIB=/opt/intel/tbb/2.0/lib/libtbb.so
TBB_LIB_FLAGS=-ltbb
TBB_INC=/opt/intel/tbb/2.0/include/

## Compilers
CC=g++
OPT_FLAGS=-O3
all:run

main-vect-vect-mult.o : main-vect-vect-mult.cc ../include/headerfiles.h ../include/ParVectVectMult.h ../include/proto.h
	$(CC) -c  main-vect-vect-mult.cc  -I$(TBB_INC)

fun_par_vector_vector_multiply.o : ../lib/fun_par_vector_vector_multiply.cc ../include/headerfiles.h ../include/ParVectVectMult.h ../include/proto.h
	$(CC) -c ../lib/fun_par_vector_vector_multiply.cc -I$(TBB_INC)

fun_vec_memory_allocation.o : ../lib/fun_vec_memory_allocation.cc ../include/headerfiles.h ../include/ParVectVectMult.h ../include/proto.h
	$(CC) -c ../lib/fun_vec_memory_allocation.cc -I$(TBB_INC)

fun_vector_input.o : ../lib/fun_vector_input.cc ../include/headerfiles.h ../include/ParVectVectMult.h ../include/proto.h
	$(CC) -c ../lib/fun_vector_input.cc -I$(TBB_INC)

fun_memoryfree.o : ../lib/fun_memoryfree.cc ../include/headerfiles.h ../include/ParVectVectMult.h ../include/proto.h
	$(CC) -c ../lib/fun_memoryfree.cc -I$(TBB_INC)

fun_print_output.o : ../lib/fun_print_output.cc ../include/headerfiles.h ../include/ParVectVectMult.h ../include/proto.h
	$(CC) -c ../lib/fun_print_output.cc -I$(TBB_INC)

run : fun_par_vector_vector_multiply.o fun_vec_memory_allocation.o fun_vector_input.o fun_memoryfree.o fun_print_output.o main-vect-vect-mult.o
	$(CC) -g  fun_par_vector_vector_multiply.o fun_vec_memory_allocation.o fun_vector_input.o fun_memoryfree.o fun_print_output.o main-vect-vect-mult.o  -o run $(TBB_LIB) 

clean:
	$(RM) -f ./run
	$(RM) -f  *.o

