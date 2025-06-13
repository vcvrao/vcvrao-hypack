SHELL=/bin/sh
BENCHMARK=jacobi
BENCHMARKU=JACOBI
include ../config/Make.inc

include ../common/make.common

OBJS = jacobi.o  ${COMMON}/print_info.o 
${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${CMPI_LIB} 

.c.o:
	${CCOMPILE} $<

jacobi.o:	jacobi.c  input_paramaters.h


clean:
	- rm -f *.o 
	- rm -f is input_paramaters.h core
