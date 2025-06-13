/*********************************************************************************

 Objective       :  Calculate the Pi Value using simple Integration as well as
                    Monte Carlo Method.

 Input           :  Number Of Sample Points
                    Number Of Threads
                    Number Of Intervals

 Output          :  Pi Value computed using simple Integration as well as
                      Monte Carlo Method.
                    Time Taken for Pi Computation(in Seconds).
                    Error Value of Pi computed using Monte Carlo Method.
                  
CLASS Details :     CLASS A 1000
                    CLASS B 10000
                    CLASS C 100000
**********************************************************************************/

#include<pthread.h>
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<sys/time.h>
#define MAX_THREADS 512
#define MAX_ITERATIONS 10000
#define Actual_pi 3.14159265388372456789123456789456   
#define tolerance 1.0E-15


#include"input_paramaters.h"
#if CLASS == 'A'
#define  CLASS_SIZE  1000 
#endif

#if CLASS == 'B'
#define  CLASS_SIZE  10000 
#endif

#if CLASS == 'C'
#define  CLASS_SIZE  100000 
#endif

void *compute_pi (void *);

double   intervalWidth, intervalMidPoint, area = 0.0;
int      numberOfIntervals, interval, iCount,iteration;
int      total_hits, total_misses, hits[MAX_THREADS],
         sample_points, sample_points_per_thread, num_threads;
double   radius=0.25,distance=0.5,four=4.0;

/* Create a MutEx for area. */

pthread_mutex_t area_mutex=PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t pi_mutex=PTHREAD_MUTEX_INITIALIZER;

void myPartOfCalc(int myID)
{

        int      myInterval;
        double   myIntervalMidPoint, myArea = 0.0, result;

        for (myInterval = myID + 1; myInterval <= numberOfIntervals; myInterval += numberOfIntervals) 
        {
        myIntervalMidPoint = ((double) myInterval - distance) * intervalWidth;
        myArea += (four / (1.0 + myIntervalMidPoint * myIntervalMidPoint));
        }

        result = myArea * intervalWidth;


        /* Lock the mutex controlling the access to area. */

        pthread_mutex_lock(&area_mutex);

        area += result;

        pthread_mutex_unlock(&area_mutex);

}

void *compute_pi (void *s) 
{
       int    seed, i, *hit_pointer;
       double rand_no_x, rand_no_y;
       int    local_hits;
 
       hit_pointer = (int *) s;
       seed = *hit_pointer;
       local_hits = 0;

       for (i = 0; i < sample_points_per_thread; i++) 
       {
          rand_no_x = (double) (rand_r(&seed))/(double)((2<<14)-1);
          rand_no_y = (double) (rand_r(&seed))/(double)((2<<14)-1);
          if (((rand_no_x - distance) * (rand_no_x - distance) +
               (rand_no_y - distance) * (rand_no_y - distance)) < radius)
          local_hits ++;
          seed *= i;
       }
      *hit_pointer = local_hits;
       pthread_exit(0) ;

}

 main ()
 {
                int i,Iteration;
                pthread_t p_threads[MAX_THREADS];
                pthread_t * threads;
                pthread_attr_t pta;
                pthread_attr_t attr;
                double computed_pi,diff;
                double time_start, time_end;
                struct timeval tv;
                struct timezone tz;
                FILE *fp;
                /* Declare a pointer to pthread to create dynamically. */

                pthread_mutex_init(&area_mutex,NULL);

                gettimeofday(&tv, &tz);
                time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;


                printf("\n\t\t---------------------------------------------------------------------------");
                printf("\n\t\t Centre for Development of Advanced Computing (C-DAC):  December-2006");
	        printf("\n\t\t C-DAC Multi Core Benchmark Suite 1.0");
                printf("\n\t\t Email : betatest@cdac.in");
	        printf("\n\t\t---------------------------------------------------------------------------");
                printf("\n\t\t Objective : PI Computations");
                printf("\n\t\t Implementation of Two different PI Computation suites");
                printf("\n\t\t Computation of PI using Numerical Integration Method ; Monte Carlo Method");

                printf("\n\t\t..........................................................................\n");
               
                 

                num_threads=THREADS;
                numberOfIntervals = num_threads;
                sample_points = CLASS_SIZE;
                printf("\n\t\t Input Parameters :");
                printf("\n\t\t CLASS %c ",CLASS );
                printf("\n\t\t Sample Points : %d",sample_points);
                printf("\n\t\t Threads       : %d ",THREADS);

                if((num_threads != 1) && (num_threads != 2) && (num_threads != 4) && (num_threads != 8))
                {
                 printf("\n Number of threads for monte carlo method should be 1 ,2,4 or 8\n");
                 exit(-1);
                }
                 if((numberOfIntervals != 1) && (numberOfIntervals != 2) && (numberOfIntervals != 4) && (numberOfIntervals != 8))
                {
                 printf("\n Number of Intervals for simple Integration method should be 1 ,2,4 or 8\n");
                 exit(-1);
                } 
   
/***********************************Simple Integration Method Starts************************************************/
                pthread_attr_init(&pta);
                 if (numberOfIntervals == 0)
                {
                 printf("\nNumber of Intervals are assumed to be 50");
                 numberOfIntervals = 50;
                }

                threads = (pthread_t *) malloc(sizeof(pthread_t) * numberOfIntervals);

                 /* Calculate Interval Width. */
                intervalWidth = 1.0 / (double) numberOfIntervals;

                /* Now Compute Area. */
                for (iCount = 0; iCount < numberOfIntervals; iCount++)
                pthread_create(&threads[iCount], &pta, (void *(*) (void *)) myPartOfCalc, (void *) iCount);

                for (iCount = 0; iCount < numberOfIntervals; iCount++)
                pthread_join(threads[iCount], NULL);

               /* Print the results. */
                pthread_attr_destroy(&pta);
/**********************************Simple Integration Method Ends*****************************************************/

/**********************************Monto Carlo Method Starts*********************************************************/
                pthread_attr_init (&attr);
                pthread_attr_setscope (&attr,PTHREAD_SCOPE_SYSTEM);
                pthread_mutex_init(&pi_mutex,NULL);
                
                iteration=0; 
                total_hits = 0;
                do
               {
                sample_points_per_thread = sample_points / num_threads;
                total_hits = 0;
                for (i=0; i< num_threads; i++) 
                {
                hits [i] = i;
                pthread_mutex_lock(&pi_mutex);
                pthread_create(&p_threads[i], &attr, compute_pi, (void *) &hits [i] ) ;
                pthread_mutex_unlock(&pi_mutex);
                }
                for (i=0; i< num_threads; i++) 
                {
                 total_hits += hits [i] ;
                 pthread_join(p_threads[i], NULL);
                }
                pthread_mutex_lock(&pi_mutex);
                computed_pi = four* (double) total_hits /((double) (sample_points));
                pthread_mutex_unlock(&pi_mutex);
                iteration++;
                pthread_attr_destroy(&attr);
               }
               while((iteration < MAX_ITERATIONS) && ((Actual_pi-computed_pi) > tolerance));
               diff=(Actual_pi-computed_pi);   
/*********************************************Monto Carlo Method Ends*************************************************/
                gettimeofday(&tv, &tz);
                time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
                pthread_attr_destroy(&attr);
                printf("\n\t\t Total number of iterations  :  %d\n", iteration);
                printf("\n\t\t Computation Of PI value Using Monte Carlo Method ........Done\n");
                printf("\n\t\t Computation Of PI value Using Numerical Integration Method ......Done\n");
                printf("\n\t\t Computed Value Of PI        :  %lf", area);
                printf("\n\t\t Time in Seconds (T)         :  %lf", time_end - time_start);
                printf("\n\t\t   ( T represents the Time taken to execute the two suites )");
                printf("\n\t\t..........................................................................\n");
                area=0.0;

    print_info( "Pi Computation ", CLASS, THREADS,time_end-time_start,
               COMPILETIME, CC, CLINK, C_LIB, C_INC, CFLAGS, CLINKFLAGS );

}
