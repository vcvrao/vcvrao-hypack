/*******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example              : counters.c

 Objective            : Usage of the PAPI High Level API's
			int PAPI_num_counters(void) 
			int PAPI_start_counters(*events, array_length)
			int PAPI_read_counters(values, array_length, check)
			int PAPI_stop_counters(values, array_length, check)

 Input                : None

 Output               : displays the total number of instructions executed
		 	executed, Total number of cycles used, 		
			and total number of TLB misses.                                                

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

*********************************************************************************/
/*
 Program Name : counters.c
 
 Description : Usage of the PAPI High Level API's
	int PAPI_num_counters(void) 
	int PAPI_start_counters(*events, array_length)
	int PAPI_read_counters(values, array_length, check)
	int PAPI_stop_counters(values, array_length, check)
 
*/

#include<stdio.h>
#include <papi.h>
#define NUM_EVENTS_MONITOR 4 

void computation_add();

int main()
	{
	  int num_hw_cntrs = 0;
	  int retval;
	  int Events[NUM_EVENTS_MONITOR] = {PAPI_TOT_INS , // Total Instructions executed
					    PAPI_TOT_CYC, //  Total Cycles used 
					    PAPI_TLB_TL,  //  Total Translation lookaside buffer misses
 					    PAPI_FP_INS,  //  Floating point instructions 	
						};
	  long_long values[NUM_EVENTS_MONITOR]; 
  	
	 
	   printf("\n  **********************************************************************************\n");	
           printf("\n\t Description : Program showing the usage of the API's \n ");
	   printf("\n\t\t PAPI_start_counters(*events, array_length) ");
	   printf("\n\t\t PAPI_read_counters(*events, array_length) ");
           printf("\n\t\t PAPI_stop_counters(values, array_length, check)\n");
	   printf("\n\t And counting the events : Total instructions executed, Total Cycles used, \n\t\t Total TLB misses, Total Floating Point instructions.\n");
	   printf("\n  **********************************************************************************\n");	
 	
	  /* Getting the number of Hardware counters on the system */
   	  printf("\n\t=> Scanning for the number of Hardware Counters supported by the current system.");
	  if ((num_hw_cntrs = PAPI_num_counters()) < PAPI_OK)
   		{
      		 printf("\n\t Error : There are no Hardware Counters available. \n");
      		 return(-1);
   		}
   	   printf("\n\t   There are %d Hardware Counters supported by this system.\n",num_hw_cntrs);

	  
	  /* Start the Hardware counters and start counting the events 
	     PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_TLB_TL, PAPI_FP_INS */
	  if ( (retval = PAPI_start_counters(Events, NUM_EVENTS_MONITOR)) != PAPI_OK)
		{
		 printf("\n\t Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
	 	 printf("\n\tError : %d %s:line %d: \n", retval,__FILE__,__LINE__); 
	 	 return(retval); }
   	  printf("\n\t=> Counters Started: Started Counting the events listed in the array.\n");
	
   	/* Do some computation */
	   computation_add();

   	 printf("\n\t=> Reading the running counter values.\n ");
	 if ( (retval = PAPI_read_counters(values, NUM_EVENTS_MONITOR)) != PAPI_OK)
		{
		 printf("\n\t Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
	 	 printf("\n\tError : %d %s:line %d: \n", retval,__FILE__,__LINE__); 
	 	 return(retval); }
   	 printf("\n\t  Summary of the Observed Events  ");

         printf("\n\t  Total Instructions executed : %lld",values[0]);	
   	 printf("\n\t  Total cycles used : %lld", values[1] );
  	 printf("\n\t  Total TLB misses : %lld ", values[2] );
   	 printf("\n\t  Total FP Instructions : %lld ", values[3] );

	  /* Stop the Hardware counters */ 
	 printf("\n\n\t=> Stop the running counters and copy the counts into array.\n");
	if ((retval=PAPI_stop_counters(values, NUM_EVENTS_MONITOR)) != PAPI_OK)
		{
		 printf("\n\t Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
	 	 printf("\n\tError : %d %s:line %d: \n", retval,__FILE__,__LINE__); 
	 	 return(retval); }
   
	return 0;
	}

void computation_add()
	{
	 int i;
	 float result;
	 printf("\n\t=> Doing some computation \n ");
	 for(i = 0;i < 1000;i++)
		{ result = result + (float) i; }
	}	
