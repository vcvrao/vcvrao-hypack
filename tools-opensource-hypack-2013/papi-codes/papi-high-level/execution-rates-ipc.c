
/*************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example            : execution-rates-ipc.c

 Objective          : Usage of the PAPI High Level API's
			int PAPI_ipc(float *rtime, float *ptime, long_long *ins, float *ipc)
 Input              : None

 Output             : Displays the total number of instructions per cycles for the 
			computation.

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

***************************************************************************/
/*
 Program Name : execution_rates_ipc.c
 
 Description : Usage of the PAPI High Level API's
	int PAPI_ipc(float *rtime, float *ptime, long_long *ins, float *ipc)
 
*/

#include<stdio.h>
#include "papi.h"

void computation_add();

int main()
	{
	  int retval;
          float real_time, proc_time, ipc;
	  long_long ins;	
          float i_real_time, i_proc_time, i_ipc;
	  long_long i_ins;	
	 
	   printf("\n  **********************************************************************************\n");	
           printf("\n\t Description : Program showing the usage of the API's \n ");
	   printf("\n\t\t PAPI_ipc(float *rtime, float *ptime, long_long *ins, float *ipc)");
           printf("\n");
	   printf("\n  **********************************************************************************\n");	
 	
          if((retval=PAPI_ipc(&i_real_time, &i_proc_time, &i_ins, &i_ipc)) < PAPI_OK)
           {
            printf("\n\t Error : Could not initialise PAPI_ipc.");
            printf("\n\t\t Your platform may not support floating point operation event.");
	    printf("\n\t Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
            return(-1);
           }

	
   	/* Do some computation */
	   computation_add();

          if((retval=PAPI_ipc(&real_time, &proc_time, &ins, &ipc)) < PAPI_OK)
           {
            printf("\n\t Error : Could not get PAPI_ipc.");
	    printf("\n\t Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
            return(-1);
           }
	   printf("\n\t Summary : ");	
           printf("\n\t Real Clock Time : %f",real_time);
           printf("\n\t Process Time : %f",proc_time);
           printf("\n\t Total number of instructions : %lld",ins);
           printf("\n\t Instruction per cycle : %f\n",ipc);


  	printf("\n"); 
	return 0;
	}

void computation_add()
	{
	 int i;
	 float result;
	 printf("\n\t=> Doing some computation \n ");
	 for(i = 0;i < 10000;i++)
		{ result = result +(float) i; }
	}	
