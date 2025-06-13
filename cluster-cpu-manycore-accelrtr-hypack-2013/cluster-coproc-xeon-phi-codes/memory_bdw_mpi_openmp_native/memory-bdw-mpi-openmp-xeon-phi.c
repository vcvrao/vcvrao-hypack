/*
*********************************************************************
               C-DAC Tech Workshop : hyPACK-2013
                       October 15-18, 2013
 
    Example     : memory-bdw-mpi-openmp-xeon-phi.c
 
   Objective    : Measure Sustained Bandwidth on Intel Xeon-Phi Sys
                   & Measure Performance
                   Bandwidth Calculation  (Simple Implementation)
                   formulation - Second Type
 
   Input        : a) Read & Write Operations  
 
   Output       : Time Elapsed and bandwidth in GBytes
 
    Created     : August-2013
 
    E-mail      : hpcfte@cdac.in
 
**********************************************************************
*/

 Input         : a) Read & Write Operations  
                      
 Description   : Implementation of Simple Bandwidth Calculation 

 Output        : Timings and Performance in Gflops/
                                                                       
 Created       : May 2013  HPC-FTE Group C-DAC, Pune 
 
 E-mail        : betatest@cdac.in {sonia, samritm, vcvrao}@cdac.in                                         

 Modified      : May  2013 

*************************************************************************/
//
//
// hellomen
//
// A simple example  that measures copy memory bandwidth on 
// on Intel Xeon Phi Co-processors Using OpenMP to scale

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#include <mpi.h>
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

#define BW_ARRAY_SIZE (1000*1000*128)
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

   int        Numprocs, MyRank;

   /* MPI initilization */

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
   MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);

	

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

   if (MyRank == 0)
   {
   	printf(" Initializing \r\n") ;
   }

   if(MyRank == 0)
   { 
   	#pragma omp parallel for 

   	for(i=0; i<BW_ARRAY_SIZE; i++)
   	{
      		fa[i] = (REAL)i + 0.1f;
      		fb[i] = (REAL)i + 0.2f;
      		fc[i] = (REAL)i + 0.2f;
   	}
   }

   /* Broadcast to every process */
   MPI_Bcast(fa,BW_ARRAY_SIZE,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(fb,BW_ARRAY_SIZE,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(fc,BW_ARRAY_SIZE,MPI_DOUBLE,0,MPI_COMM_WORLD);

   // print the # of threads to be used
   // just display from 1 thread - the "master"

   #pragma omp parallel

   if(MyRank == 0)
   {
   #pragma omp master
    printf("Starting BW Test on %d threads \r\n ",omp_get_num_threads());
   }

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
   //
   //Get the host name 
   char *hostNameArray;

   hostNameArray = (char *) malloc(sizeof(char) * 60);
   if(hostNameArray == NULL)
   {
           printf("Error: Unable to allocate memory for getHostName\n");
                exit(-1);
   }
   gethostname(hostNameArray, 256);


   if( (ttime) > 0.0f)
   {
      printf("\nprocess ID : %d hostName : %s \nGbytes\t\t:%10.3f,\nSecs\t\t:%10.3f, \nGbytes per sec\t:%10.3f \r\n",MyRank, hostNameArray, gbytes, ttime, gbytes/ttime);
   }

   free(hostNameArray);

   MPI_Finalize();
   return( 0 );
}

