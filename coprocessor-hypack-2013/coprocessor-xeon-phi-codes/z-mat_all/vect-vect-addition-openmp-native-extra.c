
/*
******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                        October 15-18, 2013

   Example    : vect-vect-addition-openmp-native.c

 Objective    : Sustained Performance for Vector-Vector Addition 
                Vector-Vector Addition (Simple Implementation)
                Formulation - Second  Type
                Execute on Intel Xeon-Phi Co-proc. & Measure Performance
                "#pragma omp parallel" 
                "#pragma offload " target(mic:MIC_DEV) 
                Quantify the performance impact of the number of 
                threads utilized per core

 Input        : a) Number of threads 
                b) size of Vector  (Size of Vector A and Vector  B) 
                c) Iterations

 Output       :  Print the Gflop/s and output Matrix C 
                   Time Elapsed and GFLOPS

 Single       : Peak Perf of Xeon -Phi : 
 Precision      1.1091 GHz X 61 cores X 16 lanes X 2 
                = 2129.6 GigaFlops/s 
                Peak Perf of Single Core = 34.90164 GigaFlop/s

 Double       : Peak Perf : 
 Precison       1.091. GHz X 61 Cores X 8 lanes X 2 
                = 1064.8 GigaFlops/s

 Created      :  August-2013

 E-mail       :  hpcfte@cdac.in     

*******************************************************************
*/
//
//
// Maximum Achievable sustaine Performance 
//
// A simple example to achieve maximum sustained  Flops 
// on Intel Xeon Phi Co-processors Using OpenMP to scale

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
//
//dtime (Wall Clock time ....)
//
//utility routine to return the current wall clock time
//
double dtime()
{
   double tseconds = 0.0;
   struct timeval mytime;
   gettimeofday(&mytime,(struct timezone*)0);
   tseconds = (double)(mytime.tv_sec +
                       mytime.tv_usec*1.0e-6);
   return( tseconds);
}


#define FLOPS_ARRAY_SIZE (1024*1024)
#define MAXFLOPS_ITERS 100000000
#define LOOP_COUNT 128
#define PEAK_PERF 2021.6
//Number of float point Operations per calculation
#define FLOPSPERCALC 2

//Define some arrays – 64 byte aligned for fast cache access
//
float Vector_A[FLOPS_ARRAY_SIZE] __attribute__((align(64)));
float Vector_B[FLOPS_ARRAY_SIZE] __attribute__((align(64)));

//
/* Main Program to Compute Gflops for different problem size(s) */
//
int main(int argc, char*argv[] )
{
   int i,j,k;
   int numthreads;

   double tstart, tstop, ttime;   
   double gflops = 0.0;
   float a = 1.1;
   double Efficiency;

   int nthreads ;
   if(argc<2)
   {
	   printf("syntax <executable> <nthreads>\n");
	   exit(1);
   }
   nthreads= atoi(argv[1]);
   //
   //initialize the compute arrays
   //

   printf(" Initializing \r\n ");


   for(i=0; i<FLOPS_ARRAY_SIZE; i++)
   {
      Vector_A[i] = (float)i + 0.1;
      Vector_B[i] = (float)i + 0.2;
   }

   printf(" Starting Compute \n " );


   /* .... OpenMP directives and API calls .... */
   /* .... Set the Number of threads for OpenMP code ...*/
   omp_set_num_threads(nthreads);

   /* ....Set the affinity to compact...... */ 
   kmp_set_defaults("KMP_AFFINITY = compact ");
   //kmp_set_defaults("KMP_AFFINITY = scatter ");
 
  #pragma omp parallel for
   for(i=0; i<FLOPS_ARRAY_SIZE; i++)
   {
      if (i==0) numthreads = omp_get_num_threads();

      Vector_A[i] = (float)i + 0.1;
      Vector_B[i] = (float)i + 0.2;
   }
   printf("starting Compute on %d threads \r\n ", numthreads);

   tstart = dtime();

   // use omp to scale the calculation
   // across the threads requested
   // need to set environment variables
   // OMP_NUM_THREADS and KMP_AFFINITY

   #pragma omp parallel for private(j,k)
   for (i=0; i<numthreads; i++)
   {
      // each thread will work its own array section 
      int offset = i*LOOP_COUNT;
   
      // loop many times to get lots of calculations
      for(j=0; j<MAXFLOPS_ITERS; j++)
      {
      //
      // scale 1st array and add in the 2nd array
      //
         for (k=0; k<LOOP_COUNT; k++)
         {
	     Vector_A[k+offset]=a*Vector_A[k+offset]+ Vector_B[k+offset];
	 }
      }
 
   }
   tstop = dtime(); 

   // No. of gigaflops we just calculated
   gflops = (double) (1.0e-9 * numthreads * LOOP_COUNT *
				MAXFLOPS_ITERS * FLOPSPERCALC);

   // elapsed time
   ttime = tstop - tstart;
  
   //
   // Print the results
   //
   if ((ttime) > 0.0f)
   {
      printf(" GFLOPS = %10.3f, Secs = %10.3f,GFLOPS per sec = %10.3f\r \n ", \
             gflops, ttime, gflops/ttime);
      printf("\nPEAK = %10.3f SUSTAINED %10.3f Efficiency %10.3f\n ", \
            PEAK_PERF, gflops/ttime, 100.0*(gflops/ttime)/PEAK_PERF );
   }
   return( 0 );
}  				


