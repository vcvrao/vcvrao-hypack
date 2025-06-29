
/********************************************************************************
 *		C-DAC Tech Workshop : hyPACK-2013
 *                           October 15-18, 2013
 *
 * Example      : pthread-findingmin-in-integerlist-mutex.c
 *    
 * Objective    : Code to find minimum value in an integer list using
 *                mutex implementation
 *      
 * Input        :  <NumElements> <NumThreads>
 * 
 * Output       : Minimum value and Execution time.
 *          
 * Description  : Here minimum value is found in an integer list using synchronisation
 *                construct Mutex.Number of elements and number of threads are taken
 *                from user.  
 *
 * Created      : August-2013
 *
 * E-mail       : hpcfte@cdac.in     
 *               
 * *********************************************************************************/


#include<pthread.h>
#include<stdio.h>
#include<sys/time.h>				// header file inclusion
#include<stdlib.h>


void checkResults(char *string, int rc);

#define MIN_INT 0				// define necessary values
#define MAXTHREADS 8

void *find_min_rwlock(void *) ;

pthread_mutex_t minimum_value_lock;

long int partial_list_size;
int minimum_value;                             // global variable declarations
long int *list;
long int NumElements;
int NumThreads;

int main (int argc,char * argv[]) 
{
 
        pthread_t       *threads;
        pthread_attr_t  pta;
        int             iteration,THREADS,ret_count;
	char 		* CLASS;

	double          time_start, time_end;
        struct          timeval tv;
        struct          timezone tz;

        int counter;        
        printf("\n\t\t Objective : To compute the minimum of a list of numbers using mutex. \n ");
       
         if( argc != 3 )                                                       // check command line arguments
	 {
	    printf("\t\t Very Few Arguments\n ");
            printf("\t\t Syntax : exec <NumElements> <Threads>\n");		
            exit(-1);
         }												// user input
         else {
            NumElements =atoi( argv[1]);
            THREADS = atoi(argv[2]);
         }

        NumThreads = THREADS;
        printf("\n\t\t Array Size  :  %ld",NumElements);
        printf("\n\t\t Threads     :  %d",NumThreads);
        printf("\n");

        if (NumThreads < 1 )
        {
           printf("\n Number of threads must be greater than zero. Aborting ...\n");
           return 0;
        }

        if ((NumThreads != 1) && (NumThreads != 2) && (NumThreads != 4) && (NumThreads != 8))
        {
           printf("\n Number of Threads must be 1 or 2 or 4 or 8. Aborting ...\n");
           return 0;
        }

        if ( ( NumElements % NumThreads ) != 0 )
        {
           printf("\n Number of threads not a factor of Integer List size. Aborting.\n");
 	   return 0 ;
        }


        partial_list_size = NumElements / NumThreads;                                 // define partial size for each thread

        list = (long int *)malloc(sizeof(long int) * NumElements);
        
         
        for(counter = 0 ; counter < NumElements ; counter++)
        {
            srand48((unsigned int)NumElements);							// random number generation
            list[counter] = (rand()%1000)+1.0;
        }
         
        threads = (pthread_t *)malloc(sizeof(pthread_t)*NumThreads);       

        minimum_value = list[0];         

  	ret_count=pthread_mutex_init(&minimum_value_lock, NULL);
        checkResults("pthread_mutex_init()\n", ret_count);
     

	ret_count=pthread_attr_init(&pta);
        checkResults("pthread_attr_init()\n", ret_count);

        pthread_attr_setscope(&pta,PTHREAD_SCOPE_SYSTEM);

	gettimeofday(&tv, &tz);
        time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

	
	for(counter = 0 ; counter < NumThreads ; counter++)
	{
           ret_count=pthread_create(&threads[counter],&pta,(void *(*) (void *)) find_min_rwlock,(void *) (counter+1));     // call find_min_rwlock function
	   checkResults("pthread_create()\n", ret_count);
	}

	gettimeofday(&tv, &tz);
        time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;


      	
        for(counter = 0 ; counter < NumThreads ; counter++)
	{
           ret_count=pthread_join(threads[counter],NULL);
	   checkResults("pthread_join()\n", ret_count);
        }
        ret_count=pthread_attr_destroy(&pta);
	checkResults("pthread_attr_destroy()\n", ret_count);

        printf("\n\t\t..........................................................................\n");
        printf("\n\t\t Minimum Value found in the Integer list     :  %d",minimum_value);
	printf("\n\t\t Time Taken in Seconds  (T)                  :  %lf Seconds",( time_end - time_start));
        printf("\n\t\t..........................................................................\n");

        
        free(list);
        free(threads);
        return 0;

 }
 void *find_min_rwlock(void * myid )                             // function to find min value
{

     int my_min;
     long int counter;

     int myId = (int)myid;

     my_min = list[(myId-1)*partial_list_size];

     for (counter = ((myId - 1) * partial_list_size); counter <= ((myId * partial_list_size) - 1); counter++){
        if (list[counter] < my_min)                                                         // each thread will operate on its own part
            my_min = list[counter];
     }

    pthread_mutex_lock(&minimum_value_lock) ;
    if (my_min < minimum_value)
    {
       minimum_value = my_min;                               // update minimum_value ( each thread will check condition)
    }
    pthread_mutex_unlock(&minimum_value_lock) ;
   
    pthread_exit(NULL);
}

void checkResults(char *string, int rc)             // check return code from pthread APIs
{
        if (rc)
  {
    printf("Error on : %s, rc=%d",string, rc);
    exit(0);
  }
  return;
}
