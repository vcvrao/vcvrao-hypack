SHELL=/bin/sh
BENCHMARK=mat_cmp_in
BENCHMARKU=mat_cmp_in
include ../config/Make.inc
include ../common/make.common

OBJS = Omp_MatMatMult_integer_V1.1.o ${COMMON}/print_info.o
${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${CMPI_LIB} 

.c.o:
	${CCOMPILE} $<

Omp_MatMatMult_integer_V1.1.o: Omp_MatMatMult_integer_V1.1.c input_paramaters.h


clean:
	- rm -f *.o 
	- rm -f is input_paramaters.h core
