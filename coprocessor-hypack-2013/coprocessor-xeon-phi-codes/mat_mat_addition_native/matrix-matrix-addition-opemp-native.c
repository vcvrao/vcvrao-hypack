/**************************************************************************************************
 *
 * 			C-DAC Tech Workshop - 2013
 * 		  	     Oct 15 - 18 , 2013
 *
 * FILE		: matrix-matrix-addtion-openmp-native.c
 *
 * INPUT	: #size of matrix , no of threads, no of iterations
 *
 * OUTPUT	: Time Elpased and GFLOPS 
 *
 * CREATED	: May,2013
 *
 * EMAIL	: hpcfte@cdac.in
 *
 **************************************************************************************************/




#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mkl.h>
#include <math.h>

void doAddition(int size,float (*restrict Matrix_A)[size], \
                            float (*restrict Matrix_B)[size], \
                            float (*restrict Matrix_C)[size])
{
	int i,j;
    // Zero the Matrix_C matrix
    #pragma omp parallel for default(none) shared(Matrix_C,size) private(i,j) 
    for ( i = 0; i < size; ++i)
        for (j = 0; j < size; ++j)
            Matrix_C[i][j] = 0.f;

// Below fragment of the code should be re-written to use numthreads in which
// each thread will compute one or many rows of the output matrix Matrix_C by
// calculating the Offset parameters.

// Compute matrix Addition.
#pragma omp parallel for default(none) \
                                      shared(Matrix_A,Matrix_B,Matrix_C,size) private(i,j)

for ( i = 0; i < size; ++i)
        for ( j = 0; j < size; ++j)
           Matrix_C[i][j] = Matrix_A[i][j] + Matrix_B[i][j];
}

/* .................... Main Program ............ */
int main(int argc, char* argv[])
{
    if(argc != 4)
    {
      fprintf(stderr,"Use: %s size nThreads nIter \n",argv[0]);
      return -1;
    }
    int i, j, k;
    int size = atoi(argv[1]);
    int nThreads = atoi(argv[2]);
    int nIter = atoi(argv[3]);

    omp_set_num_threads(nThreads);

    float (*restrict Matrix_A)[size] = malloc(sizeof(float)*size*size);
    float (*restrict Matrix_B)[size] = malloc(sizeof(float)*size*size);
    float (*restrict Matrix_C)[size] = malloc(sizeof(float)*size*size);

    // Initialize matrices
    for ( i = 0; i < size; i++) {
        for ( j = 0; j < size; j++) {
            Matrix_A[i][j] = (float)i + j;
            Matrix_B[i][j] = (float)i - j;
        }
    }


   // warm up
    doAddition (size, Matrix_A, Matrix_B, Matrix_C);

       double aveTime, minTime=1e6, maxTime=0.;
        for (i=0; i < nIter; i++) {
            double startTime = dsecnd();

            doAddition (size, Matrix_A, Matrix_B, Matrix_C);

          double endTime = dsecnd();
          double runtime = endTime - startTime;

          maxTime = (maxTime> runtime)?maxTime:runtime;
          minTime = (minTime< runtime)?minTime:runtime;
          aveTime += runtime;
     }
    aveTime /= nIter;

    printf( "%s nThrds %d matrix %d maxRT %g minRT %g aveRT %g \
            ave_GFlop/s %g\n ", argv[0], omp_get_num_threads(), size, \
            maxTime, minTime, aveTime, ((2e-9)*size)/aveTime);

   free(Matrix_A); free(Matrix_B); free(Matrix_C);
   return 0;
}
