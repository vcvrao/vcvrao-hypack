#
# *******************************************************************
#
#		C-DAC Tech Workshop : hyPACK-2013
#                           October 15-18, 2013
#
# *******************************************************************
#   Created             : August 2013
#
#   E-mail              : hpcfte@cdac.in     
#
include ./../../Make.inc

############################
# Object files
############################

OBJECTS=cpu-gpu-dgemm


##############################

$(OBJECTS): $(OBJECTS).o
	$(CC) -o $(OBJECTS) $(OBJECTS).o -L$(CBLAS_LIB) $(CBLAS_FLAGS) -L$(CUDA_LIBS) -lcublas
	mv $(OBJECTS) $(BIN)  	

$(OBJECTS).o: $(OBJECTS).cu
	$(CC_GPU) -c $(CCFLAGS) -DMKL_ILP64 -I$(CBLAS_INCLUDE) -I$(CUDA_INCLUDE) $(OBJECTS).cu


clean:
	rm -f *.o $(OBJECTS)
	rm -f  $(BIN)/$(OBJECTS)

