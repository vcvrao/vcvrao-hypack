include ../config/Make.inc


all: write_params 

# setparams creates an npbparam.h file for each benchmark 
# configuration. npbparams.h also contains info about how a benchmark
# was compiled and linked

write_params: write_params.c ../config/Make.inc
	$(UCC) ${CONVERTFLAG} -o write_params write_params.c


clean: 
	-rm -f write_params params.h
	-rm -f *~ *.o

