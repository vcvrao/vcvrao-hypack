SHELL=/bin/sh
include config/Make.inc

default:usage
pi:logo	
	cd PI; $(MAKE) THREADS=$(THREADS) CLASS=$(CLASS)
jacobi:logo	
	cd JACOBI; $(MAKE) THREADS=$(THREADS) CLASS=$(CLASS)
int_sort:logo
	cd INT_SORT; $(MAKE) THREADS=$(THREADS) CLASS=$(CLASS)
mat_cmp_in:logo	
		cd MAT_CMP_IN; $(MAKE) THREADS=$(THREADS) CLASS=$(CLASS)
mat_cmp_db:logo	
		cd MAT_CMP_DB; $(MAKE) THREADS=$(THREADS) CLASS=$(CLASS)
clean_all:logo clean
	rm -rf ./common/print_info.o ./common/write_params ./bin/* 
clean: 
	cd PI; $(MAKE) clean
	cd JACOBI; $(MAKE) clean
	cd INT_SORT; $(MAKE) clean
	cd MAT_CMP_IN; $(MAKE) clean
	cd MAT_CMP_DB; $(MAKE) clean
logo: 
	@echo "                                                                                    "
	@echo "          .........................................................................."
	@echo "            Centre for Development of Advanced Computing (C-DAC):  December-2006"
	@echo "            C-DAC Multi Core Benchmark Suite 1.0"
	@echo "            Email : betatest@cdac.in"
	@echo "          .........................................................................."
	@echo "                                                                                    "
usage:
	@echo "                                                                            "            
	@echo "         ................................................................................."
	@echo "           Usage : make <benchmark-name> THREADS=<number-of-threads> CLASS=<class-type>" 
	@echo "                   where <benchmark-name> can be pi, jacobi, mat_cmp_in, mat_cmp_db "
	@echo "                         <class-type> can be A, B, C " 
	@echo "                   View Readme.txt before using make"
	@echo "          ................................................................................"
	@echo "                                                                            "            
