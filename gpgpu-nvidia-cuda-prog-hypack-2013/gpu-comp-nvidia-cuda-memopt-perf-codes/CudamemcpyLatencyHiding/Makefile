#*******************************************************************
#
#		C-DAC Tech Workshop : hyPACK-2013
#                           October 15-18, 2013
#
# *******************************************************************
#   Created     : August 2013
#
#   E-mail      : hpcfte@cdac.in     
#
#comment	: Makefile for compiling MemcpyLatencyHiding package.
#
#*******************************************************************

#
#------------------------------------------------------
PROG = MemcpyLatencyHiding.cu
#------------------------------------------------------


# Cuda C++ compiler
CXX=nvcc 
INCLUDE = 
LIB = 

all:	release 

release: 
	$(CXX) -O2 -DNDEBUG $(CXXFLAGS) -o run $(PROG) -I$(INCLUDE) -L$(LIB) 
clean:
	rm -f run *.o *.d output.txt
test:
	./$(PROG) 1000
