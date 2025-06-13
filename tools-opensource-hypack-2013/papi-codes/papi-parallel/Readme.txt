
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
# *****************************************************
#
TITLE : PAPI High Level API's
AUTHOR : Shiva 
DATE : Jan 02 2012


Programs Description 
 1. Program Name : papi-pthreads-find-min-value.c
    Description : The program randomly generated the list of integers.The list
		  of integers is partitioned equally among the threads. The 
		  size of each threads partition of stored in a variable and
		  the pointer to the start of each threads partial list is
		  passed to it as a pointer.The minimum value is protected by 
		  the mutex-lock .Threads execute the mutex lock to gain
		  exclusive access to the minimum value. Once this access is 
		  gained, the value is updated as required, and the lock 
		  subsequently released. Since at any time, only one thread 
		  can hold the lock, only one thread can update the value. As
		  each thread does the work, an event set is created with the
		  events PAPI_TOT_INS, PAPI_TOT_CYC, & PAPI_LST_INS and these 
		  events data for each thread are captured. 

Compilation :
	For compiling the codes, use the Makefile provided with make utility.
