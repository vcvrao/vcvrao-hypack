
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
# ***************************************************
#
## Please update the PAPI_INC and PAPI_LIB
## according to the installation on the system.
PAPI_INC = /usr/local/papi-c-3.9/include/
PAPI_LIB = /usr/local/papi-c-3.9/lib/libpapi.a
CC = gcc
CFLAGS += -I$(PAPI_INC)
LDFLAGS =  $(PAPI_LIB)

TARGETS = avail-num-counters 


all: $(TARGETS) 

$(TARGETS):  %:%.o 
	$(CC) -o $@ $(CFLAGS) $^  $(LDFLAGS)
clean:
	$(RM) *.o $(TARGETS) 

