/*
******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                        October 15-18, 2013

 Example      : openmp4x-vect-vect-addition-perf.c

 Objective    : Sustained Performance for Vector-Vector Addition 
                Vector-Vector Addition (Sequential Implementation)
                Formulation - Second  Type
                Execute on Intel Xeon-Phi Co-proc. & Measure Performance
                Quantify the performance 

 Input        : a) size of Vector  (Size of Vector A and Vector  B) 
                b) Iterations

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

*******************************************************************/
//
//
// A simple example to try  to get lots of Flops on Intel Xeon 
// Phi Co-processors.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#define SIZE 1000
#pragma omp declare target
#define FLOPS_ARRAY_SIZE (1024*1024)
#define MAXFLOPS_ITERS 1000
#define LOOP_COUNT 128

//Floating pt ops per inner loop iteration
#define FLOPSPERCALC 2

//
//dtime (Wall Clock time ....)
//
//utility routine to return
//the current wall clock time
//

int My_saxpy(float *Vector_A,float *Vector_B)
{

  float a=1.1;
  #pragma omp target map(Vector_A[0:FLOPS_ARRAY_SIZE]) map(Vector_B[0:FLOPS_ARRAY_SIZE]) map(a)
  {
    for(int i=0;i<FLOPS_ARRAY_SIZE;i++)
	    Vector_A[i] = a * Vector_A[i] + Vector_B[i];
  }
}
double dtime()
{
   double tseconds = 0.0f;
   struct timeval mytime;
   gettimeofday(&mytime,(struct timezone*)0);
   tseconds = (double)(mytime.tv_sec +
                       mytime.tv_usec*1.0e-6);
   return( tseconds);
}



//Define some arrays : 64 byte aligned for fast cache access
float Vector_A[FLOPS_ARRAY_SIZE] __attribute__((align(64)));
float Vector_B[FLOPS_ARRAY_SIZE] __attribute__((align(64)));

//
/* Main Program to Compute Gflops for different problem size(s) */
//
int main(int argc, char*argv[] )
{
   int i,j,k;
   double tstart, tstop, ttime;   
   double gflops = 0.0;
   float a = 1.1;

   //
   //initialize the compute arrays
   //

   printf("Initializing \r\n ");

   for(i=0; i<FLOPS_ARRAY_SIZE; i++)
   {
      Vector_A[i] = (float)i + 0.1;
      Vector_B[i] = (float)i + 0.2;
   }

   printf(" Starting Compute \n" );

   tstart = dtime();
   /* loop many times to really get lots of calculations */


   for(j=0; j<MAXFLOPS_ITERS; j++)
   {
   
    //
    // scale 1st array and add in the 2nd array
    //
       // for (k=0; k<LOOP_COUNT; k++)
       // {
         //    Vector_A[k] = a * Vector_A[k] + Vector_B[k];
        //}
	My_saxpy(Vector_A,Vector_B);
   }
   tstop = dtime(); 

   // No. of gigaflops we just calculated
   gflops = (double) (1.0e-9 * LOOP_COUNT *
				MAXFLOPS_ITERS * FLOPSPERCALC);

   // elapsed time
   ttime = tstop - tstart;
  
   //
   // Print the results
   //
   if ((ttime) > 0.0f)
   {
      printf(" GFLOPS = %10.5f, Secs = %10.5f, GFLOPS per sec = %10.5f \r \n ", gflops, ttime, gflops/ttime);
   }
   return( 0 );
}  				


