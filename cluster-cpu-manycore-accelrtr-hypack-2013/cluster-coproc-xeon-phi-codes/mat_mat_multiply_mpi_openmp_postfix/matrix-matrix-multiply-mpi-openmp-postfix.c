/*
***********************************************************************
		C-DAC Tech Workshop : hyPACK-2013
                        October 15-18, 2013

   Example 	     :  matrix-matrix-multiply-mpi-openmp-postfix.c

   Objective         :  To implement Matrix Matrix multiplication 
                        Algorithm using openMP on Xeon Phi Coprocessor

   Input             :  Automatic input generation  of Input Matrix data 
                        Size of the Square Matrices 

   Output            :  Print the Gflop/s and output Matrix C 
                        Time Elapsed and GFLOPS

   Created           :  August-2013

   E-mail            :  hpcfte@cdac.in     

************************************************************************
*/

#ifndef MIC_DEV
#define MIC_DEV 0
#endif

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <assert.h>
#include <mkl.h>
#include <mpi.h>

// An OpenMP simple matrix multiply 
void doMult(int size,float (* restrict Matrix_A)[size], \
                     float (* restrict Matrix_B)[size], \
                     float (* restrict Matrix_C)[size]) 
{ 

      	#pragma omp parallel for default(none) shared(Matrix_C,size) 
      	// Zero the C matrix 
      	for(int i = 0; i < size; ++i) 
      	for(int j = 0; j < size; ++j)  
        Matrix_C[i][j] =0.f; 

     	// Compute matrix multiplication. 
     	#pragma omp parallel for default(none) shared(Matrix_A,Matrix_B,Matrix_C,size) 
     	for (int i = 0; i < size; ++i) 
     	for (int k = 0; k < size; ++k) 
     	for (int j = 0; j < size; ++j) 
      	Matrix_C[i][j] += Matrix_A[i][k] * Matrix_B[k][j]; 
}
/* ................... Main Program ................ Starts ...........*/

int main(int argc, char* argv[]) 
{ 
	/* check number of args */
  	if(argc != 4) 
  	{ 
   		fprintf(stderr,"Use: %s size nThreads nIter \n",argv[0]); 
   		return -1; 
  	}


  	int        Numprocs, MyRank;

 	/* ........MPI Initialisation ....... */

  	MPI_Init(&argc, &argv);
  	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
  	MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);


	/* variable declarations */
  	/* for getting host name */
	int namelen;
	char name[MPI_MAX_PROCESSOR_NAME];
 	
	/* for communication part */
	int tag=1, index;
 	MPI_Status status;
 	char msg[6] = "hello";
 	char inmsg[6] = "";
  	
	/* for matrix multiplication */
	int  i,j,k; 
  	
	int  size     = atoi(argv[1]); 
  	int  nThreads = atoi(argv[2]); 
  	int  nIter    = atoi(argv[3]); 
	int dsec00 = 0;
  	double aveTime, minTime = 1e6, maxTime =0.0; 


	/* communication between Procs : begin */
 	if (MyRank == 0)
 	{
		for(index = 1; index < Numprocs ; index++)
		{
 			MPI_Send(&msg,6,MPI_CHAR,index,tag,MPI_COMM_WORLD); 
			printf("\nMessage send by rank = %d to rank = %d is %s\n", MyRank, index, msg);
		}
 	}
 	else
 	{
 		MPI_Recv(&inmsg,6,MPI_CHAR,0,tag,MPI_COMM_WORLD,&status);
		printf("\nMessage recieved by rank = %d is %s\n", MyRank, inmsg);
 	}

	/* communication between Procs : Ends */
 


	/* create nThreads number of threads */
  	omp_set_num_threads(omp_get_max_threads()); 
	printf("\nMKL threads = %d\n", omp_get_max_threads());
	/* memory allocation */
  	float (*restrict Matrix_A)[size] = malloc(sizeof(float)*size*size); 
  	float (*restrict Matrix_B)[size] = malloc(sizeof(float)*size*size); 
  	float (*restrict Matrix_C)[size] = malloc(sizeof(float)*size*size); 

  	// Fill the A and B arrays 
   	#pragma omp parallel for default(none) shared(Matrix_A, Matrix_B,size) \
                            private(i,j,k) 
   	for(i = 0; i < size; ++i) 
	{ 
       		for(j = 0; j < size; ++j) 
		{ 
	         	Matrix_A[i][j] = (float)i + j; 
         		Matrix_B[i][j] = (float)i - j; 
    		} 
  	}

  	// warm up 
  	doMult(size, Matrix_A,Matrix_B,Matrix_C); 
    	
	/* Multiplication */
	for (int i=0; i < nIter; i++) 
	{ 
     		double startTime = dsecnd(); 

      		// Matrix Multiplication 
    		doMult(size, Matrix_A,Matrix_B, Matrix_C); 

     		double endTime = dsecnd(); 
     		double runtime = endTime - startTime; 

     		maxTime = (maxTime > runtime)?maxTime:runtime; 
     		minTime = (minTime < runtime)?minTime:runtime; 

    		aveTime += runtime; 
   	} 
   	aveTime /= nIter; 

	/* get host name: begins */
   	MPI_Get_processor_name (name, &namelen);
	/* get host name: begins */
   	printf("\nProcess %d is running on %s host with results : \n%s nThrds %d matrix %d maxRT %g minRT %g aveRT %g ave_GFlop/s %g\n", MyRank, name,argv[0],omp_get_num_threads(), size, maxTime, minTime, aveTime, 2e-9*size*size*size/aveTime); 
  
	/* freeing all memories */
	free(Matrix_A); 
  	free(Matrix_B); 
  	free(Matrix_C); 
  
 	MPI_Finalize();
  	return 0; 
}
/* ..........Program Listing Completed ............. */


