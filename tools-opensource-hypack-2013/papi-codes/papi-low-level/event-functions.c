/*****************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013


 Example              : event-functions.c


 Objective            :  Program to create event set, add events to event set, starting,
             		 reading, adding, removing & stopping events in an event set, 
	       		 emptying & destroying event set. 

		       Functions used are 
			int PAPI_create_eventset(int *EventSet)			
			int PAPI_add_event(int *EventSet, int Event)
			int PAPI_start(int EventSet)
			int PAPI_read(int *EventSet, long_long *values)
			int PAPI_accum(int EventSet, long_long *values)
			int PAPI_remove_event(int EventSet, int Event)
			int PAPI_cleanup_eventset(EventSet)			
 			int PAPI_destroy_eventset(EventSet)	

 Input                : None

 Output               : Displays the hardware informatin (CPU info)


   Created            : August-2013

   E-mail             : hpcfte@cdac.in     

*****************************************************************************/

/*
 Program Name : event-functions.c
 
 Description : Program to create event set, add events to event set, starting,
               reading, adding, removing & stopping events in an event set, 
	       emptying & destroying event set. 
	       Functions used are 
		int PAPI_create_eventset(int *EventSet)			
		int PAPI_add_event(int *EventSet, int Event)
		int PAPI_start(int EventSet)
		int PAPI_read(int *EventSet, long_long *values)
		int PAPI_accum(int EventSet, long_long *values)
		int PAPI_remove_event(int EventSet, int Event)
		int PAPI_cleanup_eventset(EventSet)			
 		int PAPI_destroy_eventset(EventSet)		

*/

#include<stdio.h>
#include<stdlib.h>
#include "papi.h"
#define TOTAL_EVENTS 18

int** my_generate_matrix(int, int**);
int** my_matrix_multiply(int, int **, int **, int **);
void matrix_matrix_multiply(void);

int main()
	{
          // Declare the variables for the event functions
	 int EventSet = PAPI_NULL;
	 int Events[TOTAL_EVENTS];
	 long_long values[TOTAL_EVENTS];
         int retval;
	 char *prg_disscription = "Program to create event set, start events, read the events \n\t\t"\
		"values, add events to the event set, remove the events from the \n\t\t "\
		"event set, clean the event set and destroy the event set."\
	       "\n\t\t Functions used are: "\
		"\n\t\t\t int PAPI_create_eventset(int *EventSet)"\
		"\n\t\t\t int PAPI_add_event(int *EventSet, int Event)"\
		"\n\t\t\t int PAPI_start(int EventSet)"\
		"\n\t\t\t int PAPI_read(int *EventSet, long_long *values)"\
		"\n\t\t\t int PAPI_remove_event(int EventSet, int Event)"\
		"\n\t\t\t int PAPI_cleanup_eventset(EventSet)"\
 		"\n\t\t\t int PAPI_destroy_eventset(EventSet)";
	   printf("\n  **********************************************************************************\n");	
	  printf("\n\t Description : %s",prg_disscription);
           printf("\n");
	   printf("\n  **********************************************************************************\n");	
 	
      	  printf("\n\t=> PAPI Library initialization : ");
	  if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
   	   {
      	    printf("\n\t  Error : PAPI Library initialization error! \n");
     	    return(-1);
	   }
      	  printf("Done");

 	  /* Create the event set*/
      	  printf("\n\t=> Creating Eventset : ");
	  if((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)	
	    {	printf("\n\t  Error : PAPI failed to create the Eventset\n");	
        	printf("\n\t  Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
     	        return(-1);
	    }
      	  printf("Done");

	  /* Adding events : 
			PAPI_TOT_INS    Instructions completed
			PAPI_TOT_CYC    Total cycles
			PAPI_LST_INS    Load/store instructions completed
			PAPI_L2_TCM     Level 2 cache misses
			PAPI_TLB_TL     Total translation lookaside buffer misses
			*/
      	  printf("\n\t=> Adding events to the created eventset : ");
      	  printf("\n\t\t Adding event PAPI_TOT_INS");
	  if((retval = PAPI_add_event(EventSet, PAPI_TOT_INS)) != PAPI_OK)	
	    {	printf("\n\t Error : PAPI failed to add event (PAPI_TOT_INS)\n");	
        	printf("\n\t Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval); }
      	  printf("\n\t\t Adding event PAPI_TOT_CYC");
	  if((retval = PAPI_add_event(EventSet, PAPI_TOT_CYC)) != PAPI_OK)	
	    {	printf("\n\t Error : PAPI failed to add event (PAPI_TOT_CYC)\n");	
        	printf("\n\t Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval); }
      	  printf("\n\t\t Adding event PAPI_LST_INS");
	  if((retval = PAPI_add_event(EventSet, PAPI_LST_INS)) != PAPI_OK)	
	    {	printf("\n\t Error : PAPI failed to add event (PAPI_LST_INS)\n");	
        	printf("\n\t Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval); }
      	  printf("\n\t\t Adding event PAPI_L2_TCM");
	  if((retval = PAPI_add_event(EventSet, PAPI_L2_TCM)) != PAPI_OK)	
	    {	printf("\n\t   Error : PAPI failed to add event (PAPI_L2_TCM)\n");	
        	printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval); }
      	  printf("\n\t\t Adding event PAPI_TLB_TL");
	  if((retval = PAPI_add_event(EventSet, PAPI_TLB_TL)) != PAPI_OK)	
	    {	printf("\n\t   Error : PAPI failed to add event (PAPI_TLB_TL)\n");	
        	printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval); }

	  /* Starting the events of the event set */	
	  printf("\n\t=> Starting the events of eventset : ");
	  if((retval = PAPI_start(EventSet)) != PAPI_OK)	
	    {	printf("\n\t   Error : PAPI failed to start the events in created Eventset\n");	
        	printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
     	        return(-1);    }
      	  printf("Done");

            /* Doing some computation here (square matrix-matrix multiplication )... */
      	    printf("\n\t=> doing matrix-matrix multiplication ");
	    matrix_matrix_multiply();

	  /* Reading the values of the events defined in the event set*/  
	  printf("\n\t=> Reading the values from events of the eventset : ");
	  if((retval = PAPI_read(EventSet, values)) != PAPI_OK)	
	    {	printf("\n\t   Error : PAPI failed to copy counters value in created Eventset to values array\n");	
        	printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
     	        return(-1);   }
      	  printf("Done");

	 printf("\n\t=> Summary of the values read from the events of eventset: ");
	 printf("\n\t  Total # Instructions : %ld",values[0]);
	 printf("\n\t  Total # Cycles : %ld",values[1]);
	 printf("\n\t  Total # Load/Store Instructions : %ld",values[2]);
	 printf("\n\t  Total # L2 cache misses : %ld",values[3]);
	 printf("\n\t  Total # TLB misses : %ld",values[4]);

	 /* Stop the created event set */	
	  printf("\n\t=> Stop the events of the eventset");
	  if((retval = PAPI_stop(EventSet, values)) != PAPI_OK)	
	    {	printf("\n\t   Error : PAPI failed to stop the events in created Eventset");	
        	printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
     	        return(-1);  }
      	  printf("Done");

	 /* Remove the event from event set */	
	  printf("\n\t=> Remove the event (PAPI_TLB_TL) from the eventset");
	  if((retval = PAPI_remove_event(EventSet, PAPI_TLB_TL)) != PAPI_OK)	
	    {	printf("\n\t   Error : PAPI failed to remove the event (PAPI_TLB_TL) from created Eventset");	
        	printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
     	        return(-1);  }
      	  printf("Done");
	    
	
	 /* Clean up the events of the event set */	
	  printf("\n\t=> Clean the events from the eventset");
	  if((retval = PAPI_cleanup_eventset(EventSet)) != PAPI_OK)	
	    {	printf("\n\t   Error : PAPI failed to clean the events from created Eventset");	
        	printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
     	        return(-1);  }
      	  printf("Done");

	 /* Clean up the events of the event set */	
	  printf("\n\t=> Deleting the eventset");
	  if((retval = PAPI_destroy_eventset(&EventSet)) != PAPI_OK)	
	    {	printf("\n\t   Error : PAPI failed to clean the events from created Eventset");	
        	printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
     	        return(-1); }
      	  printf("Done");
	
	PAPI_shutdown();
   	printf("\n-----------------------------------------------------------------------------\n");
	return 0;
	}

/*
 Function : matrix_matrix_multiply()
 Author: Shiva  Date : Dec 13 2008
*/


void matrix_matrix_multiply(void)
 {
  int irow,icol,i;
  int mat_size;	
  int **mat_a,**mat_b,**mat_c;
  srand((unsigned)time(NULL)); 
  mat_size = 100;
  printf("\n\t   Size of matrix taken as %d",mat_size);

    /* Memory Allocation for Matrix A */
    printf("\n\t   Generating the Matrix A and Matrix B with size %d.",mat_size);	
    mat_a = my_generate_matrix(mat_size, mat_a);
    mat_b = my_generate_matrix(mat_size, mat_b);
    
    /* Matrix Multiplication  Matrix C = Matrix A * Matrix B*/
    printf("\n\t   Matrix multiplication : ");	
    mat_c = my_matrix_multiply(mat_size, mat_a, mat_b, mat_c);
    printf(" Done ");	

  /* Freeing the Memory ... */
  free(mat_a);  free(mat_b);  free(mat_c);  } 


int** my_generate_matrix(int size, int **mat)
    {     int irow,icol,i;
    /* Memory Allocation for Matrix and populating matrix with randomly generated number */
    mat = (int **)malloc(size * sizeof(int));
    for(irow = 0; irow < size; irow++)
    	{  	mat[irow] = (int *)malloc(size * sizeof(int));	
     	  for(icol = 0; icol < size; icol++)
      		{      mat[irow][icol] = rand()%10;		}    	}
	return mat;    }

int** my_matrix_multiply(int size, int **mat_a, int **mat_b, int **mat_c)
	{     	int irow,icol,i;
        /* Allocating the memory for the resultant */ 
    	mat_c = (int **)malloc(size * sizeof(int));
	    for(irow = 0; irow < size; irow++)
    		{ mat_c[irow] = (int *)malloc(size * sizeof(int));	
     		for(icol = 0; icol < size; icol++)
      			{      mat_c[irow][icol] = 0;		}  	}
	 /* Matrix Multiplication  Matrix C = Matrix A * Matrix B*/
	for(irow = 0; irow < size; irow++)
	        {     	for(icol = 0; icol < size; icol++)
                	{ for(i=0;i<size;i++)
        	             {  mat_c[irow][icol] = mat_c[irow][icol] + (mat_a[irow][i]*mat_b[i][icol]);  } } } 
	return mat_c; }



