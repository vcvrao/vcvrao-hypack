SHELL=/bin/sh
BENCHMARK=mat_cmp_in
BENCHMARKU=mat_cmp_in
include ../config/Make.inc
include ../common/make.common

OBJS = Pthreads_Matrix_Comp_Integer-V1.0.o ${COMMON}/print_info.o
${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${CMPI_LIB} 

.c.o:
	${CCOMPILE} $<

Pthreads_Matrix_Comp_Integer-V1.2.o: Pthreads_Matrix_Comp_Integer-V1.0.c input_paramaters.h


clean:
	- rm -f *.o 
	- rm -f is input_paramaters.h core
