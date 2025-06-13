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
TITLE : PAPI High Level API's

#

Programs Description 
 1. Program Name : initilize-papi.c
    Description : Program to initilize PAPI library and shutdown
                  PAPI Library (Free the resources used by the PAPI).
                  The API's PAPI_library_init(version) and PAPI_shutdown(void)
		  are used.	

 2. Program Name : event-functions.c 
    Description : Program to create event set, start events, read the events
		  values, add events to the event set, remove the events
		  from the event set, clean the event set and destroy the 
		  event set.

 3. Program Name : cpu-info.c 
    Description : Programs to get the cpu info.

 4. Program Name : executable-info.c
    Description : Program to get the executable info such as start and end
		  address of the text and data segment.

 Note :	For compiling the codes, use the Makefile provided with make utility.
