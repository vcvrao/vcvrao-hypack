SHELL=/bin/sh
BENCHMARK=int_sort
BENCHMARKU=INT_SORT
include ../config/Make.inc
include ../common/make.common

OBJS = Sorting_Minimum_value-V1.0.o  ${COMMON}/print_info.o 
${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${CMPI_LIB} 

.c.o:
	${CCOMPILE} $<

Sorting_Minimum_value-V1.0.o:	Sorting_Minimum_value-V1.0.c  input_paramaters.h


clean:
	- rm -f *.o 
	- rm -f is input_paramaters.h core
