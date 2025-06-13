/**************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example              : initilize-papi.c

 Objective            :  Program to initilize papi and Shutdown PAPI. 
		         PAPI_library_init(version)
	   	         PAPI_shutdown(void);

 Input                : None

 Output               : Displays the hardware informatin (CPU info)


   Created            : August-2013

   E-mail             : hpcfte@cdac.in     

*****************************************************************************/

/*
 Program Name :initilze-papi.c
 
 Description : Program to initilize papi and Shutdown PAPI. 
	        PAPI_library_init(version)
	   	PAPI_shutdown(void);
 
*/

#include<stdio.h>
#include "papi.h"


int main()
	{
	  int retval;
	   printf("\n  **********************************************************************************\n");	
           printf("\n\t Description : Program to initilize PAPI library and shutdown \n\t\t PAPI Library (Free the resources used by the PAPI) \n ");
	   printf("\n\t\t PAPI_library_init(version)");
	   printf("\n\t\t PAPI_shutdown(void)");
           printf("\n");
	   printf("\n  **********************************************************************************\n");	
 	

	  if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
   	   {
      	    printf("\n\t Error : PAPI Library initialization error! \n");
     	    return(-1);
	   }

      	    printf("\n\t Greetings !!! : PAPI Library initialization Sucess !! \n");

	  /*if((retval = PAPI_shutdown()) != PAPI_OK )
   	   {
      	    printf("\n\t Error : PAPI Failed to shutdown ! \n");
     	    return(-1);
	   }*/

	   PAPI_shutdown();
      	   printf("\n\t Greetings !!! : PAPI Shutdown Sucess !! \n");

  	printf("\n"); 
	return 0;
	}

