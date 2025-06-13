SHELL=/bin/sh
BENCHMARK=pi
BENCHMARKU=PI
include ../config/Make.inc
include ../common/make.common

OBJS = Omp_Pi_Comp_v1.0.o  ${COMMON}/print_info.o 
${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${CMPI_LIB} 

.c.o:
	${CCOMPILE} $<

Omp_Pi_Comp_v1.0.o:	Omp_Pi_Comp_v1.0.c  input_paramaters.h


clean:
	- rm -f *.o 
	- rm -f is input_paramaters.h core
