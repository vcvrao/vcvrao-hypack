
/***************************************************************************
		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example               : pthread-mutex.c

 Objective             : To perform dot product of vector using mutex.

 Input                 : None.

 Output                : Dot product of two vectors 	                                            
                                                                        
 Created      : August-2013

 E-mail       : hpcfte@cdac.in     

*****************************************************************************/



#include <pthread.h>
#include <stdio.h>
#include <malloc.h>

#define NUMTHRDS 4
#define VECLEN 100

/*   
The following structure contains the necessary information  
to allow the function "dotprod" to access its input data and 
place its output into the structure.  
*/

typedef struct 
 {
   double      *a;
   double      *b;
   double     sum; 
   int     veclen; 
 } DOTDATA;

/* Define globally accessible variables and a mutex */

 DOTDATA dotstr; 
 pthread_t callThd[NUMTHRDS];
 pthread_mutex_t mutexsum;

/*
The function dotprod is activated when the thread is created.
All input to this routine is obtained from a structure 
of type DOTDATA and all output from this function is written into
this structure. The benefit of this approach is apparent for the 
multi-threaded program: when a thread is created we pass a single
argument to the activated function - typically this argument
is a thread number. All  the other information required by the 
function is accessed from the globally accessible structure. 
*/

void *dotprod(void *arg)
{

   /* Define and use local variables for convenience */

   int i, start, end, offset, len ;
   double mysum, *x, *y;
   offset = (int)arg;
     
   len = dotstr.veclen;
   start = offset*len;
   end   = start + len;
   x = dotstr.a;
   y = dotstr.b;

   /*
   Perform the dot product and assign result
   to the appropriate variable in the structure. 
   */

   mysum = 0;
   for (i=start; i<end ; i++) 
    {
      mysum += (x[i] * y[i]);
    }

   /*
   Lock a mutex prior to updating the value in the shared
   structure, and unlock it upon updating.
   */
   pthread_mutex_lock (&mutexsum);
   dotstr.sum += mysum;
   pthread_mutex_unlock (&mutexsum);

   pthread_exit((void*) 0);
}

/* 
The main program creates threads which do all the work and then 
print out result upon completion. Before creating the threads,
the input data is created. Since all threads update a shared structure, 
we need a mutex for mutual exclusion. The main thread needs to wait for
all threads to complete, it waits for each one of the threads. We specify
a thread attribute value that allow the main thread to join with the
threads it creates. Note also that we free up handles when they are
no longer needed.
*/

int main (int argc, char *argv[])
{
   int i;
   double *a, *b;
   int status;
   pthread_attr_t attr;
   int ret_count;


  printf("\n\t\t---------------------------------------------------------------------------");
  printf("\n\t\t Centre for Development of Advanced Computing (C-DAC)");
  printf("\n\t\t Email : betatest@cdac.in");
  printf("\n\t\t---------------------------------------------------------------------------");
  printf("\n\t\t Objective : To perform dot product of vector using Mutex.\n ");
  printf("\n\t\t..........................................................................\n");


   /* Assign storage and initialize values */
   a = (double*) malloc (NUMTHRDS*VECLEN*sizeof(double));
   b = (double*) malloc (NUMTHRDS*VECLEN*sizeof(double));
  
   for (i=0; i<VECLEN*NUMTHRDS; i++)
    {
     a[i]=1.0;
     b[i]=a[i];
    }

   dotstr.veclen = VECLEN; 
   dotstr.a = a; 
   dotstr.b = b; 
   dotstr.sum=0;

   ret_count=pthread_mutex_init(&mutexsum, NULL);
   if (ret_count)
   {
         printf("ERROR; return code from pthread_mutex_init() is %d\n", ret_count);
         exit(-1);
   }
         
   /* Create threads to perform the dotproduct  */
   ret_count=pthread_attr_init(&attr);
   if (ret_count)
   {
       printf("ERROR; return code from pthread_attr_init() is %d\n", ret_count);
       exit(-1);
   }

   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(i=0; i<NUMTHRDS; i++)
        {
	/* 
	Each thread works on a different set of data.
	The offset is specified by 'i'. The size of
	the data for each thread is indicated by VECLEN.
	*/
	    ret_count=pthread_create( &callThd[i], &attr, dotprod, (void *)i);
      	    if (ret_count)
      	    {
         	printf("ERROR; return code from pthread_create() is %d\n", ret_count);
         	exit(-1);
      	    }
	}

 	ret_count=pthread_attr_destroy(&attr);
        if(ret_count)
	{
		printf("\n ERROR : return code from pthread_attr_destroy() is %d\n",ret_count);
		exit(-1);
	}
                


        /* Wait on the other threads */
	/*for(i=0; i<NUMTHRDS; i++)
        {
	  ret_count=pthread_join( callThd[i], (void **)&status);
	  if (ret_count)
      	  {
         	printf("ERROR; return code from pthread_join() is %d\n", ret_count);
         	exit(-1);
      	  }
	
	}*/



	for(i=0; i<NUMTHRDS; i++)
		pthread_join( callThd[i], (void **)&status);

   /* After joining, print out the results and cleanup */
   printf ("Sum =  %f \n", dotstr.sum);
   free (a);
   free (b);
   ret_count=pthread_mutex_destroy(&mutexsum);
   if (ret_count)
   {
         printf("ERROR; return code from pthread_mutex_destroy() is %d\n", ret_count);
         exit(-1);
   }
   pthread_exit(NULL);
}   
