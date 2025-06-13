/***************************************************************************************************
* FILE		: openmp4x-mat-mat-addition.c
*
* INPUT		: Nil
*
* OUTPUT	: Displays Host and device output matrices
*
* CREATED	: August,2013
*
* EMAIL		: hpcfte@cdac.in
*
***************************************************************************************************/

#include <stdio.h>

 #define SIZE 4
 #pragma omp declare target

int Mat_Mat_Add(int(*Matrix_A)[SIZE], \
                int(*Matrix_B)[SIZE], \
                int(*Matrix_Device)[SIZE]) 
		
{
  #pragma omp target map(Matrix_A[0:SIZE][0:SIZE]) \
                     map(Matrix_B[0:SIZE][0:SIZE]) \
		     map(Matrix_Device[0:SIZE][0:SIZE])
  {
    for(int i=0;i<SIZE;i++)
    	for(int j=0;j<SIZE;j++)
    Matrix_Device[i][j] = Matrix_A[i][j] + Matrix_B[i][j];
  }
//  return sum;
}

int main()
{
  int Matrix_A[SIZE][SIZE], \
      Matrix_B[SIZE][SIZE], \
      Matrix_Host[SIZE][SIZE], \
      Matrix_Device[SIZE][SIZE]; 

  for(int i=0; i<SIZE; i++){
  	for(int j=0; j<SIZE; j++){
  Matrix_A[i][j]=i+j;
  Matrix_B[i][j]=i+j;
  Matrix_Host[i][j]=0.0;
  Matrix_Device[i][j]=0.0;
  }
  }

  for(int i=0;i<SIZE;i++)
     for(int j=0;j<SIZE;j++)
       Matrix_Host[i][j] = Matrix_A[i][j] + Matrix_B[i][j];

  printf("Host Matrix\n");
  for(int i=0; i<SIZE; i++){
	  for(int j=0; j<SIZE; j++){
		  printf("%d ",Matrix_Host[i][j]);
	  }
	  printf("\n");
  }
			    
  Mat_Mat_Add(Matrix_A,Matrix_B,Matrix_Device);

  printf("Device Matrix\n");
  for(int i=0; i<SIZE; i++){
	  for(int j=0; j<SIZE; j++){
		  printf("%d ",Matrix_Device[i][j]);
	  }
	  printf("\n");
  }
}
