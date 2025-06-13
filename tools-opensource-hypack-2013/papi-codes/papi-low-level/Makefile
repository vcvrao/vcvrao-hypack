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

## Makefile for the programs using PAPI Low-level API
## Update the PAPI_LIB & PAPI_INC according to the 
## PAPI Instatllation of the system.
PAPI_INC = /usr/local/papi-c-3.9/include/
PAPI_LIB = /usr/local/papi-c-3.9/lib/libpapi.a
CC = gcc
CFLAGS += -I$(PAPI_INC)
LDFLAGS =  $(PAPI_LIB)

TARGETS = event-functions executable-info cpu-info initilize-papi 


all: $(TARGETS) 

$(TARGETS):  %:%.o 
	$(CC) -o $@ $(CFLAGS) $^  $(LDFLAGS)
clean:
	$(RM) *.o $(TARGETS) 

