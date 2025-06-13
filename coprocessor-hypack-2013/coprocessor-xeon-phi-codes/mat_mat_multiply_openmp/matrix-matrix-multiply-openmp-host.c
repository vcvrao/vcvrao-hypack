
/*
*********************************************************************
		C-DAC Tech Workshop : hyPACK-2013
                        October 15-18, 2013

   Example 	     :  matrix-matrix-multiply-openmp-host.c

   Objective         :  To implement Matrix Matrix multiplication 
                        Algorithm using openMP on Xeon Phi Coprocessor
                        "#pragma offload " target(mic:MIC_DEV) 

   Input             : a) Number of threads 
   	               b) size of Square Matrix(i.e Size of Matrix A and Matrix B) 
                       c) Iterations

   Output            :  Print the Gflop/s and output Matrix C 
                        Time Elapsed and GFLOPS

   Created           :  August-2013

   E-mail            :  hpcfte@cdac.in     

**********************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <assert.h>
#include <mkl.h>

// An OpenMP simple matrix multiply 
void doMult(int size,float (* restrict Matrix_A) [size], float (*restrict Matrix_B)[size],  float (*restrict Matrix_C)[size]) 
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
  if(argc != 4) 
  { 
   fprintf(stderr,"Use: %s size nThreads nIter \n",argv[0]); 
   return -1; 
  } 

  int  i,j,k; 
  int  size     = atoi(argv[1]); 
  int  nThreads = atoi(argv[2]); 
  int  nIter    = atoi(argv[3]); 

  omp_set_num_threads(nThreads); 

  float (*restrict  Matrix_A)[size]  = malloc(sizeof(float)*size*size); 
  float (*restrict Matrix_B)[size] = malloc(sizeof(float)*size*size); 
  float (*restrict Matrix_C)[size] = malloc(sizeof(float)*size*size); 

  // Fill the A and B arrays 
   #pragma omp parallel for default(none) shared(Matrix_A, Matrix_B,size) \
                            private(i,j,k) 
   for(i = 0; i < size; ++i) { 
       for(j = 0; j < size; ++j) { 
         Matrix_A[i][j] = (float)i + j; 
         Matrix_B[i][j] = (float)i - j; 
    } 
  }

  // warm up 
  doMult(size, Matrix_A,Matrix_B,Matrix_C); 

  double aveTime, minTime = 1e6, maxTime =0.0; 
    for (int i=0; i < nIter; i++) { 
     double startTime = dsecnd(); 

      // Matrix Multiplication 
      doMult(size, Matrix_A, Matrix_B, Matrix_C); 

     double endTime = dsecnd(); 
     double runtime = endTime - startTime; 

     maxTime =(maxTime> runtime)?maxTime:runtime; 
     minTime =(minTime< runtime)?minTime:runtime; 

    aveTime += runtime; 
   } 
   aveTime /= nIter; 

   printf("%s nThrds %d matrix %d maxRT %g minRT %g aveRT %g  ave_GFlop/s %g\n", 
             argv[0], omp_get_num_threads(), size, 
             maxTime, minTime, aveTime, 2e-9*size*size*size/aveTime); 
  free(Matrix_A); 
  free(Matrix_B); 
  free(Matrix_C); 

  return 0; 
}
/* ..........Program Listing Completed ............. */


