/*************************************************************************
        Objective           :  PI calculation using OpenMP Parallel For
                              Directive. 

   *************************************************************************/


#include<stdio.h>
#include<omp.h>
#include <sys/time.h>
#include <math.h>
#define PI 3.14159265388372456789123456789456
#define MAX_ITERATIONS 10000
#define TOLERANCE 1.0E-15

#include"input_paramaters.h"
#if CLASS == 'A'
#define  CLASS_SIZE  100
#endif

#if CLASS == 'B'
#define  CLASS_SIZE  1000
#endif

#if CLASS == 'C'
#define  CLASS_SIZE  10000
#endif


double walltime(double *); 

main()
{
	long int       Noofintervals;
	int            Noofthreads,threadnum[4];
	int            i,threadid,reexe,i_red,Iteration;
        double         x, totalsum, h,error,finalsum,error_red,x_red;
	int            count,local_count,t,tid,iteration ;
	unsigned short seed[3];
        double         x_cor,y_cor;
        double         distance = 0.5,radius_square = 0.25,four = 4;
        FILE           *fp;
	struct timeval  TimeValue_Start;
        struct timezone TimeZone_Start;

        struct timeval  TimeValue_Final;
        struct timezone TimeZone_Final;
        long            time_start, time_end;
        double          time_overhead;
	

    
        printf("\n\t\t---------------------------------------------------------------------------");
        printf("\n\t\t Centre for Development of Advanced Computing (C-DAC):  December-2006");
        printf("\n\t\t C-DAC Multi Core Benchmark Suite 1.0");
        printf("\n\t\t Email : betatest@cdac.in");
        printf("\n\t\t---------------------------------------------------------------------------");
        printf("\n\t\t Objective : PI Computations");
        printf("\n\t\t Implementation of Two different PI Computation suites");
        printf("\n\t\t Computation of PI using Numerical Integration Method ; Monte Carlo Method");

        printf("\n\t\t..........................................................................\n");
           Noofthreads = THREADS;
           Noofintervals = CLASS_SIZE;
           printf("\n\t\t Sample Points :  %ld",Noofintervals);
           printf("\n\t\t Threads       :  %d",Noofthreads);

            totalsum = 0.0;
            count = 0;
            iteration = 0;
            gettimeofday(&TimeValue_Start, &TimeZone_Start);
            time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
 /* ....iterations for convergence of Pi computation...................*/            
         do
         {
            h = 1.0 / Noofintervals;
       	   // Noofthreads = threadnum[reexe];
      	    omp_set_num_threads(Noofthreads);
            #pragma omp parallel for private(x)
            for (i = 1; i < Noofintervals + 1; i = i+1) {
                 x = h * (i - distance);
            #pragma omp critical
                 totalsum = totalsum + four/(1.0 + x * x);
            }
            totalsum = totalsum * h;
            iteration++;
            //printf("First Method  - Iteration : %d \tFirst Method - Computed PI Value   : %3.25lf \n",iteration,totalsum);
         }while((iteration < MAX_ITERATIONS) &&((PI-totalsum) < TOLERANCE));
            printf("\n\t\t Computation of PI value using Numerical Integration Method (Critical Section).....Done\n");
  	   // printf(" Calculated PI first method   : %1.15lf \n",totalsum);
           

     /*   ......Second computation starts...........      */
         iteration = 0;
         finalsum = 0.0;
         do
         {
   	    #pragma omp parallel for private(x_red) reduction(+:finalsum)
            for (i_red = 1; i_red < Noofintervals + 1; i_red = i_red+1) {
                  x_red = h * (i_red - distance);
                  finalsum = finalsum + four/(1.0 + x_red * x_red);
            }

            finalsum = finalsum * h;
            error_red = fabs(PI-finalsum);
            iteration++;
            //printf("2 - Iteration : %d\n",iteration);
         }while((iteration < MAX_ITERATIONS) &&(error_red < TOLERANCE));
           //printf(" Numerical Integration Method(Reduction Clause)\n");
            printf("\n\t\t Computation of PI value using Numerical Integration Method (Reduction Clause).....Done\n");
	   // printf(" Calculated  PI  second method : %1.15lf  \n",finalsum);
           // printf(" Error                         : %1.16lf\n",  error_red);

/*           Second Pie Computation ..............is over ...*/

/*          Third Method PIe computation..........         */
         iteration = 0;
         do{
  	    #pragma omp parallel private(seed,t,i,x_cor,y_cor,local_count)
            {
                local_count = 0;
		seed[0] = 0.0;
		seed[1] = 1.0;
                seed[2] = tid = omp_get_thread_num();
                t = omp_get_num_threads();
                /*printf("\t\t Number of threads %d\n",t);*/
                for(i = tid; i<Noofintervals;i+=t)
                {
			x_cor = erand48(seed);
			y_cor = erand48(seed);
                        if (((x_cor - distance) * (x_cor - distance) +
               (y_cor - distance) * (y_cor - distance )) <= radius_square)
                                local_count++;
                }
            #pragma omp critical
                count += local_count ;
               
              } iteration ++;
          }while((iteration < MAX_ITERATIONS) &&((PI -(4.0*count/Noofintervals)) < TOLERANCE));
	    gettimeofday(&TimeValue_Final, &TimeZone_Final);
            time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
           // printf("\n Monte Carlo Method \n");
            printf("\n\t\t Computation Of PI value Using Monte Carlo Method ........Done\n");
           // printf(" Calculated PI third method : %1.15lf  \n",4.0*count/Noofintervals);
           // printf(" Error         : %1.16lf\n", fabs(PI-(4.0*count/Noofintervals)));

        time_overhead = (time_end - time_start)/1000000.0;
  	printf("\n\t\t Calculated PI               : %1.15lf \n",totalsum);
       // printf("\n\t\t Error                       : %1.16lf\n",fabs(PI-totalsum));
        printf("\n\t\t Time in Seconds (T)         :  %lf",time_overhead);
        printf("\n\t\t   ( T represents the Time taken to execute the two suites )");
        printf("\n\t\t..........................................................................\n");

  print_info( "Pi Computation ", CLASS, THREADS,time_overhead,
               COMPILETIME, CC, CLINK, C_LIB, C_INC, CFLAGS, CLINKFLAGS  );


}
	
