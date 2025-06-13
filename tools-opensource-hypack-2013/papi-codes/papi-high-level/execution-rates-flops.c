
/*************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example            : execution-rates-flops.c

 Objective          : Usage of the PAPI High Level API's
	              int PAPI_flops(float *rtime, float *ptime, long_long *flpops, float *mflops)

 Input              : None

 Output             : Displays the total number of flops of computation.



   Created          : August-2013

   E-mail          : hpcfte@cdac.in     

*****************************************************************************/
/*
 Program Name : execution-rates-flops.c
 
 Description : Usage of the PAPI High Level API's
	int PAPI_flops(float *rtime, float *ptime, long_long *flpops, float *mflops)
 
*/

#include<stdio.h>
#include "papi.h"

void computation_add();

int main()
	{
	  int retval;
          float real_time, proc_time, mflops;
	  long_long flp_ops;	
          float ireal_time, iproc_time, imflops;
	  long_long iflp_ops;	
	 
	   printf("\n  **********************************************************************************\n");	
           printf("\n\t Description : Program showing the usage of the API's \n ");
	   printf("\n\t\t PAPI_flops(float *rtime, float *ptime, long_long *flpops, float *mflops)");
           printf("\n");
	   //printf("\n\t And counting the events : Total instructions executed, Total Cycles used, \n\t\t Total TLB misses, Total Floating Point instructions.\n");
	   printf("\n  **********************************************************************************\n");	
 	
          if((retval=PAPI_flops(&ireal_time, &iproc_time, &iflp_ops, &imflops)) < PAPI_OK)
           {
            printf("\n\t Error : Could not initialise PAPI_flops.");
            printf("\n\t\t Your platform may not support floating point operation event.");
            printf("\n\t Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
            return(-1);
           }

	
   	/* Do some computation */
	   computation_add();

          if((retval=PAPI_flops(&real_time, &proc_time, &flp_ops, &mflops)) < PAPI_OK)
           {
            printf("\n\t Error : Could not get PAPI_flops.");
            printf("\n\t Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
            return(-1);
           }
	   printf("\n\t Summary : ");	
           printf("\n\t Real Clock Time : %f",real_time);
           printf("\n\t Process Time : %f",proc_time);
           printf("\n\t Number of floating point arithmetic operations : %lld",flp_ops);
           printf("\n\t MFlops(or Millions of Floating point operations per second) rate : %f\n",mflops);


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
