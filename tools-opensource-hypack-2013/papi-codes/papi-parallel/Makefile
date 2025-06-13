
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
# *******************************************************************
#
## Makefile for the parallel programs with PAPI for hands-on
## Update the PAPI_INC & PAPI_LIB according to the PAPI Installation.
PAPI_INC = /usr/local/papi-c-3.9/include/
PAPI_LIB = /usr/local/papi-c-3.9/lib/libpapi.a
CC = gcc
CFLAGS += -I$(PAPI_INC)
LDFLAGS =  $(PAPI_LIB) -lpthread

TARGETS = papi-pthreads-find-min-value


all: $(TARGETS) 

$(TARGETS):  %:%.o 
	$(CC) -o $@ $(CFLAGS) $^  $(LDFLAGS)
clean:
	$(RM) *.o $(TARGETS) 

