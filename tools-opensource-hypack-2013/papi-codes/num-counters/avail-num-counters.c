/********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example              : avail-num-counters.c

 Objective            : PAPI_num_counters returns the number of 
			hardware counters the platform has. 
			Returns negative number if there is 
                	an error.    

 Input                : None

 Output               : Prints the number of available hardware 				
			counters.                                           
                                                                        
   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

*******************************************************************/

/*
 Program Name : avail_num_counters.c
 
 Descriptoin :  PAPI_num_counters returns the number of 
		hardware counters the platform has. 
		Returns negative number if there is 
                an error.                          	 
*/

#include<stdio.h>
#include "papi.h"

int main()
	{
	  int num_hwcntrs = 0;
	  if ((num_hwcntrs = PAPI_num_counters()) < PAPI_OK)
   		{
      		 printf("\n\t Error : There are no counters available. \n");
      		 return(-1);
   		}
   	   printf("\n\t There are %d counters in this system.\n",num_hwcntrs);
	return 0;
	}

