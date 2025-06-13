/*
*********************************************************************
                 C-DAC Tech Workshop : hyPACK-2013
                         October 15-18, 2013
 
    Example     : memory-bdw-openmp-xeon-phi.c
 
   Objective    : Measure Sustained Bandwidth on Intel Xeon-Phi Sys
                   & Measure Performance
                   Bandwidth Calculation  (Simple Implementation)
                   formulation - Second Type
 
   Input        : a) Read & Write Operations  
 
   Output       : Time Elapsed and bandwidth in GBytes
 
    Created     : August-2013
 
    E-mail      : hpcfte@cdac.in
 
************************************************************************
*/
//
//
//----------------------------------------------------------------  		
//  		:  Processor Xeon Phi 5110P  :
//----------------------------------------------------------------  		
// 1. co-processor version capacity 8GB : processor Xeon Phi 5110P
// 2. memory channel interface speed : 5.0 Giga Transfer/ Sec ( GT/s)
// 3. 8 memory controller , each accessing two memory channels , 
// used on co-processor.
// 4. each memory transaction to GDDR5 memory is 4 bytes of data,
//    resulting in 5.0 GT/s * 4 bytes or 20 GB/s per channel.
// 5. 16 Total channel provide maximul transfer rate of 320 GB/s.
// 6. achieved bandwidth is 40% of the peak
// 7. how to tap the memory bandwidth, another key performance of 
//    Intel Xeon Phi.
// 8. effective bandwidth in the range of 50 to 60% of peak memory 
//    bandwidth can be achieved with some optimizatioin.
//
// A simple example  that measures copy memory bandwidth on 
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

// Set to float or double
#define REAL double

#define BW_ARRAY_SIZE (1000*1000*64)
#define BW_ITERS 1000

// number of men ops each iteration
// 1 read + 1 write
#define OPS_PERITER 2

// define some arrays
// make sure they are 64 byte aligned - for 
// fastest cache access

REAL fa[BW_ARRAY_SIZE] __attribute__((align(64)));
REAL fb[BW_ARRAY_SIZE] __attribute__((align(64)));
REAL fc[BW_ARRAY_SIZE] __attribute__((align(64)));

//
// --------------------------------------------------
/* Main Program to Compute Bandwidth  */
//
int main(int argc, char*argv[] )
{
   int i,j,k;
   int numthreads;

   double tstart, tstop, ttime;
   double gbytes = 0.0f;
   REAL a = 1.1f;
   //
   // Initialize the compute arrays
   //

   //double tstart, tstop, ttime;   
   double gflops = 0.0f;
   //float a = 1.1f;

   //
   //Initialize the compute arrays
   //

   printf(" Initializing \r\n") ;

   #pragma omp parallel for 

   for(i=0; i<BW_ARRAY_SIZE; i++)
   {
      fa[i] = (REAL)i + 0.1f;
      fb[i] = (REAL)i + 0.2f;
      fc[i] = (REAL)i + 0.2f;
   }

   // print the # of threads to be used
   // just display from 1 thread - the "master"

   #pragma omp parallel

   #pragma omp master
    printf("Starting BW Test on %d threads \r\n ",omp_get_num_threads());

   tstart = dtime();

   // use omp to scale the test across
   // the threads requested. Need to set environment
   // variable OMP_NUM_THREADS and KMP_AFFINITY

   for (i=0; i<BW_ITERS; i++)
   {
     //
     // copy the arrays to/from memory (2 bw ops)
     // use openmp to scale and get aggregate bw
     //

     #pragma omp parallel for
        for(k=0; k<BW_ARRAY_SIZE; k++)
        {
	   fa[k] = fb[k];
	}
    }

   tstop = dtime(); 
 
   // # of gigabytes we just moved

   gbytes = (double)( 1.0e-9 * OPS_PERITER * BW_ITERS *
                      BW_ARRAY_SIZE*sizeof(REAL));
   
   // elapsed time
   ttime = tstop - tstart;
   //
   // Print the results
   //
   if( (ttime) > 0.0f)
   {
      printf("Gbytes = %10.31f,Secs = %10.31f," "Gbytes per sec - %10.31f \r\n ",
                gbytes, ttime, gbytes/ttime);
   }
   return( 0 );
}

