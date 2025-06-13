/*************************************************************************

 Objective  :  Finds the Minimum Value in the Integer List. 

 Input      :  Number of Threads,
         
               Integer List Size.

 Output     :  Minimum value in the Integer List

*************************************************************************/

#include<pthread.h>
#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>
#define MIN_INT 0

#include"input_paramaters.h"
#if CLASS == 'A'
#define NumElements 1000 
#endif

#if CLASS == 'B'
#define NumElements  10000
#endif

#if CLASS == 'C'
#define NumElements  100000
#endif

void *find_min(void *) ;

pthread_mutex_t minimum_value_lock;

long int partial_list_size;
int minimum_value;
long int *list;
int NumThreads;

int main (int argc,char * argv[]) {
 
        pthread_t       *threads;
        pthread_attr_t  pta;
        int             iteration;
        double          time_start, time_end;
        struct          timeval tv;
        struct          timezone tz;
        double          MemoryUsed = 0.0;
        FILE            *fp; 
        char            filename[40];
        int counter,i;        
        printf("\n\t\t---------------------------------------------------------------------------");
        printf("\n\t\t Centre for Development of Advanced Computing (C-DAC):  December-2006");
        printf("\n\t\t C-DAC Multi Core Benchmark Suite 1.0");
        printf("\n\t\t Email : betatest@cdac.in");
        printf("\n\t\t---------------------------------------------------------------------------");
        printf("\n\t\t Objective : Sorting Single Dimension Array (Integer Operations)\n ");
        printf("\n\t\t Performance of Sorting a Minimum value in a large Single Dimension Array ");
        printf("\n\t\t on Multi Socket Multi Core Processor using 1/2/4/8 threads \n");
        printf("\n\t\t Input Parameters :");
        printf("\n\t\t..........................................................................\n");
        
        NumThreads = THREADS;
        printf("\n\t\t Array Size  :  %d",NumElements);
        printf("\n\t\t Threads     :  %d",NumThreads);
        printf("\n");


        partial_list_size = NumElements / NumThreads;

        list = (long int *)malloc(sizeof(long int) * NumElements);
        MemoryUsed += ( NumElements * sizeof(long int));
          
        gettimeofday(&tv, &tz);
        time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

        sprintf(filename,"./../In-House-Pthreads-Benchmarks/INT_SORT/Data/int_sort.%d",NumElements);
        if ((fp = fopen (filename, "r")) == NULL)
           {
           printf("\nCan't open input file");
           exit(-1);
           }
           printf("\n Opening input file %s",filename);
           for (i = 0; i < NumElements; i++)
           fscanf(fp, "%ld ",&list[i]);
           fclose(fp);
 
        threads = (pthread_t *)malloc(sizeof(pthread_t)*NumThreads);       

        minimum_value = list[0];         

        pthread_mutex_init(&minimum_value_lock, NULL);
         
        pthread_attr_init(&pta);

        pthread_attr_setscope(&pta,PTHREAD_SCOPE_SYSTEM);

        for(counter = 0 ; counter < NumThreads ; counter++)
           pthread_create(&threads[counter],&pta,(void *(*) (void *)) find_min,(void *) (counter+1));
      
        for(counter = 0 ; counter < NumThreads ; counter++)
           pthread_join(threads[counter],NULL);
     
        pthread_attr_destroy(&pta);

        gettimeofday(&tv, &tz);
        time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

        printf("\n\t\t Minimum Value found in the Integer list     :  %d",minimum_value);
        printf("\n\t\t Memory Utilised                             :  %lf MB",(MemoryUsed / (1024*1024)));
        printf("\n\t\t Time Taken in Seconds  (T)                  :  %lf Seconds",( time_end - time_start));   
        printf("\n\t\t   ( T represents the Time taken to  Minimum Value )\n");
        printf("\n\t\t..........................................................................\n");
        print_info( "Integer Sorting", CLASS, THREADS,time_end-time_start,
               COMPILETIME, CC, CLINK, C_LIB, C_INC, CFLAGS, CLINKFLAGS );
       

        
        free(list);
        free(threads);
        return 0;

 }
 void *find_min(void * myid ) {

     int my_min;
     long int counter;

     int myId = (int)myid;

     my_min = list[(myId-1)*partial_list_size];

     for (counter = ((myId - 1) * partial_list_size); counter <= ((myId * partial_list_size) - 1); counter++){
        if (list[counter] < my_min)
            my_min = list[counter];
     }

     /* lock the mutex associated with minimum_value and update the variable as      
     required */

    pthread_mutex_lock(&minimum_value_lock) ;
    if (my_min < minimum_value)
       minimum_value = my_min;
 
    /* and unlock the mutex */
    pthread_mutex_unlock(&minimum_value_lock) ;
   
    pthread_exit(NULL);
}

