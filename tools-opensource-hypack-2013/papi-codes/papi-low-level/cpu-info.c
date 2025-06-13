/*****************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example              : cpu-info.c


 Objective            : Program to get the hardware info with the
	       		const PAPI_hw_info_t *PAPI_get_hardware_info(void);



	 typedef struct _papi_hw_info {
      	int ncpu;                    		 // Number of CPU's in an SMP Node 
      	int nnodes;                  		 // Number of Nodes in the entire system 
      	int totalcpus;               		 // Total number of CPU's in the entire system 
      	int vendor;                  		 // Vendor number of CPU
      	char vendor_string[PAPI_MAX_STR_LEN];      // Vendor string of CPU 
      	int model;                 		 // Model number of CPU 
      	char model_string[PAPI_MAX_STR_LEN];       // Model string of CPU 
      	float revision;               		 // Revision of CPU 
      	float mhz;                   		 // Cycle time of this CPU 
      	PAPI_mh_info_t mem_hierarchy;  		 // PAPI memory heirarchy description 
   	} 	PAPI_hw_info_t;

 Input                : None

 Output               : Displays the hardware informatin (CPU info)

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

*****************************************************************************/


/*
 Program Name : cpu-info.c
 
 Description : Program to get the hardware info with the
	       const PAPI_hw_info_t *PAPI_get_hardware_info(void);

 typedef struct _papi_hw_info {
      int ncpu;                    		 // Number of CPU's in an SMP Node 
      int nnodes;                  		 // Number of Nodes in the entire system 
      int totalcpus;               		 // Total number of CPU's in the entire system 
      int vendor;                  		 // Vendor number of CPU
      char vendor_string[PAPI_MAX_STR_LEN];      // Vendor string of CPU 
      int model;                 		 // Model number of CPU 
      char model_string[PAPI_MAX_STR_LEN];       // Model string of CPU 
      float revision;               		 // Revision of CPU 
      float mhz;                   		 // Cycle time of this CPU 
      PAPI_mh_info_t mem_hierarchy;  		 // PAPI memory heirarchy description 
   } PAPI_hw_info_t;


 
*/

#include<stdio.h>
#include "papi.h"

void computation_add();

int main()
	{
	  int retval;
	  const PAPI_hw_info_t *hw_info = NULL;
	   printf("\n  **********************************************************************************\n");	
           printf("\n\t Description : Program to get the Hardware (CPU) Info using the API  ");
	   printf("\n\t\t  const PAPI_hw_info_t *PAPI_get_hardware_info(void)");
           printf("\n");
	   printf("\n  **********************************************************************************\n");	
 	

	  if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
   	   {
      	    printf("\n\t Error : PAPI Library initialization error! \n");
     	    return(-1);
	   }

	  if ((hw_info = PAPI_get_hardware_info()) == NULL)
   	   {
            printf("\n\t Error : Could not get hardware info .");
	    return(-1);	
	   }

	   printf("\n\t Hardware Info (CPU)  ");	
	   printf("\n\t CPU Vendor : %s", hw_info->vendor_string);
           printf("\n\t CPU model string : %s", hw_info->model_string);
	   printf("\n\t CPU Vendor ID : %d", hw_info->vendor);
	   printf("\n\t CPU model ID : %d", hw_info->model);
	   printf("\n\t CPU revision : %f", hw_info->revision);
 	   printf("\n\t Total # of CPU: %d ",hw_info->totalcpus);
 	   printf("\n\t CPU Frequency : %f ",hw_info->mhz);

	   PAPI_shutdown();

  	printf("\n"); 
	return 0;
	}

