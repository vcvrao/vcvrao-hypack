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

TITLE : PAPI High Level API's
DATE : Jan 02 2008


Programs Description 
 1. Program Name : counters.c
    Description : Program to start counting values into the values array,
		  Read the running counter values, and stop the running counters. 

 2. avail-num-counters.c
    Description : Program to get the optimal length of the values array for
	          the high level functions.	

 3. Program Name : timers.c
    Description : Programs using the PAPI Real and Virtual Timers.

 4. Program Name : execution-rates-flops.c
    Description : Program to measure the Floating operations executed and
		  the MFlops rating.

 5. Program Name : execution-rates-ipc.c
    Description : Program to measure the toal instructions execurted  and
		  the Instructions per cycle.

Compilation :
	For compiling the codes, use the Makefile provided with make utility.
