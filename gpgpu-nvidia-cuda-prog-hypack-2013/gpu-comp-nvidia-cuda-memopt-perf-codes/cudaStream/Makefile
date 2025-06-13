##---------- Makefile for opencl codes----------------
##---------- uncomment the necessary object to compile----------------



#CLHOME=/usr/local/cuda
#CLINCLUDE=$(CLHOME)/include
#CLLIB=$(CLHOME)/lib64


CLHOME=
CLINCLUDE=
CLLIB=
CC = nvcc
CCFLAGS = -arch sm_20



# Names of set of example programs

OBJECT = streamdp

run:
	$(CC) $(CCFLAGS) $(CLLINKFLAGS) -I$(CLINCLUDE) -L$(CLLIB) -o $(OBJECT) $(OBJECT).cu

clean:
	rm -f $(OBJECT)

