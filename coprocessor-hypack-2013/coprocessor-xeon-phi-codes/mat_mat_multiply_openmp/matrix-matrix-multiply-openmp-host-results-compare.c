
/*
***********************************************************************
		C-DAC Tech Workshop : hyPACK-2013
                        October 15-18, 2013

   Example 	   :  matrix-matrix-multply-openmp-host-results-compare.c

   Objective       : MIC-OpenMP program to demonstrate the use of OpenMP 
                     "#pragma omp parallel" for default(none) shared(A,B,C,size) 
                     "#pragma offload " target(mic:MIC_DEV) 
                     Quantify the performance impact of the number of 
                     threads utilized per core and address scaling ....

   Input           : a) Number of threads 
 		     b) size of Square Matrix(i.e Size of Matrix A and Matrix B) 

   Output          :  Print the Gflop/s and output Matrix C 
                      Time Elapsed and GFLOPS

   Created         :  August-2013

   E-mail          :  hpcfte@cdac.in     

************************************************************************
*/
#
#ifndef MIC_DEV
#define MIC_DEV 0
#endif

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <assert.h>
#include <mkl.h>

// An OpenMP simple matrix multiply 
void doMult(int size,float (* restrict Matrix_A)[size], \
                     float (* restrict Matrix_B)[size], \
                     float (* restrict Matrix_C)[size]) 
{ 
/*
   #pragma offload target(mic:MIC_DEV) \
      in(Matrix_A:length(size*size)) \
      in(Matrix_B:length(size*size)) \
     out(Matrix_C:length(size*size))
*/

   {
      // Zero the C matrix 
      #pragma omp parallel for default(none) shared(Matrix_C,size) 
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
}
float nrmsdError(int size, float (* restrict M1)[size], \ 
                           float (* restrict M2)[size])  
{     
  double sum = 0.; 
  double max,min; 
  max = min = (M1[0][0]- M2[0][0]);   

  #pragma omp parallel for     
  for (int i = 0; i < size; ++i)   
   for (int j = 0; j < size; ++j) {     
     double diff = (M1[i][j]- M2[i][j]); 

  #pragma omp critical     
      {       
         max = (max > diff)?max:diff;     
         min = (min < diff)?min:diff;   
         sum += diff*diff;    
      }  
    } 
     return(sqrt(sum/(size*size))/(max-min)); 
}

float doCheck(int size, float (* restrict Matrix_A)[size],   
                        float (* restrict Matrix_B)[size],  
                        float (* restrict Matrix_C)[size],      
                        int   nIter,     
                        float *error)  
{   

       float (*restrict MatrixTranspose_A)[size] = malloc(sizeof(float)*size*size);   
       float (*restrict MatrixTranspose_B)[size] = malloc(sizeof(float)*size*size);   
       float (*restrict MatrixTranspose_C)[size] = malloc(sizeof(float)*size*size);   
       float (*restrict Cgemm)[size] = malloc(sizeof(float)*size*size);  

        // transpose to get best sgemm performance 

       #pragma omp parallel for   
          for(int i = 0; i < size; i++)    
          for(int j = 0; j < size; j++) {     
             MatrixTranspose_A[i][j] = Matrix_A[j][i];      
             MatrixTranspose_B[i][j] = Matrix_B[j][i];   
          }     

         float alpha = 1.0f, beta = 0.0f;  /* Scaling factors */  

          // warm up   
          sgemm("N", "N", &size, &size, &size, &alpha,     \
               (float *)MatrixTranspose_A, &size,  \
               (float *)MatrixTranspose_B, &size, &beta,  \
               (float *)MatrixTranspose_C, &size);  

          double mklStartTime = dsecnd();   
    
           for(int i=0; i < nIter; i++)    

           sgemm("N", "N", &size, &size, &size, &alpha,  
                (float *)MatrixTranspose_A, &size, \
                (float *)MatrixTranspose_B, &size, &beta, \
                (float *)MatrixTranspose_C, &size);   

          double mklEndTime = dsecnd();     

            // transpose in Cgemm to calculate error  

            #pragma omp parallel for  
              for(int i=0; i < size; i++)     
                 for(int j=0; j < size; j++)    
                    Cgemm[i][j] = MatrixTranspose_C[j][i];   

              *error = nrmsdError(size, MatrixTranspose_C, Cgemm);   
                  
            free(MatrixTranspose_A); 
            free(MatrixTranspose_B); 
            free(MatrixTranspose_C); 
            free(Cgemm);   
     
         return (2e-9*size*size*size/((mklEndTime-mklStartTime)/nIter) );    
}


/* ................... Main Program ................ Starts ...........*/

int main(int argc, char* argv[]) 
{ 
  if(argc != 4) 
  { 
   fprintf(stderr,"Use: %s size nThreads nIter \n",argv[0]); 
   return -1; 
  } 

   int i,j,k;
   int  size     = atoi(argv[1]); 
   int  nThreads = atoi(argv[2]); 
   int  nIter    = atoi(argv[3]); 

   omp_set_num_threads(nThreads); 

   float (*restrict Matrix_A)[size] = malloc(sizeof(float)*size*size); 
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

  double aveDoMultTime = 0.;  
 {     
   // warm up     
    doMult(size, Matrix_A, Matrix_B, Matrix_C);   

    double startTime = dsecnd();     
 
    for(i=0; i < nIter; i++) { 
       doMult(size, Matrix_A, Matrix_B, Matrix_C);     
    }    

    double endTime = dsecnd();    
    aveDoMultTime = (endTime - startTime)/nIter;   
  } 

 #pragma omp parallel 
 #pragma omp master   
     printf("%s nThreads %d matrix %d %d runtime %g GFlop/s %g", \
                argv[0], omp_get_num_threads(), size, size,  \
               aveDoMultTime, 2e-9*size*size*size/aveDoMultTime); 

 #pragma omp barrier   

  // do check   
  float error = 0.f;   
  float mklGflop = doCheck(size, Matrix_A,Matrix_B,Matrix_C,nIter,&error);  
  printf(" mklGflop %g NRMSD_error %g", mklGflop, error);    

  printf("\n");     

  free(Matrix_A); 
  free(Matrix_B); 
  free(Matrix_C); 

  return 0; 
}
/* ..........Program Listing Completed ............. */

