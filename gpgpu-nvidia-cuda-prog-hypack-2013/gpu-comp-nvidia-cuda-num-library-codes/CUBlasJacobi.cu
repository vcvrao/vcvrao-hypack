/****************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Objective   : program to solve a  linear system of equations (Ax = b) using jacobi 
                method in a GPU (using cublas lib)

  Input       : Number of unknowns and maximum number of iterations

  Output      : Solution Vector.

  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

****************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<cuda.h>
#include "cublas.h"
//-------------------------------------------------------------------------------------------------------------------------
#define BLOCKSIZE 16 
#define ERR 0
#define DOMINANCE 1
//-------------------------------------------------------------------------------------------------------------------------
void   InitializeVectors(float** coeffMat, float** solutionVect, float** rhsVect, float** diagOfCoeffMat, float** tempSolution, int Size, double *diag);
void   generateCoeffMat(int Size, float** coeffMat, double *diag);
void   generateRhsVect(int Size, float** rhsVect, float* coeffMat);
void   generateSolutionVect(int Size, float** solutionVect);
void   generateDiagOfCoeffMat(int Size, float** diagOfCoeffMat, float* coeffMat);
void   generateTempSolution(float** tempSolutioni,int);
float  calcDiff(float* solutionVect, float* tempSolution, int Size);
void   outputSolution(float* solutionVect, int iteration, double timing, int Size);
void   freeHostMemory(float* coeffMat, float* solutionVect, float* rhsVect, float* diagOfCoeffMat, float* tempSolution);
void*  malloc_safe_call(int size);
//-------------------------------------------------------------------------------------------------------------------------

/************************************************************************************
*  pragma routine to report the detail of cuda error
************************************************************************************/
#define CUDA_SAFE_CALL(call)                                                    \
            do{                                                                 \
                cudaError_t err = call;                                         \
                if(err != cudaSuccess)                                          \
                 {                                                              \
                   fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                   __FILE__, __LINE__, cudaGetErrorString( err) );              \
                   exit(1);                                                     \
                 }                                                              \
             } while (0)                                                        \

//---------------------------------------------------------------------------------------------------------------------------
//kernel that performs parallel division of scalar pairs which are given as corresponding components of two vectors.
__global__ void ParallelScalarDiv( float* vector1, float* vector2, float *result, int vectDim, int threadDim)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int tindex = (threadDim * tidx) + tidy;
    	int maxNumThread = threadDim * threadDim;
	int pass = 0;  
	int threadColumnIdx;
	while( (threadColumnIdx = (tindex + maxNumThread * pass))  < vectDim )
	{
		result[threadColumnIdx] = vector1[threadColumnIdx] / vector2[threadColumnIdx];            
		pass++;
	}
	__syncthreads();
}
//---------------------------------------------------------------------------------------------------------------------------

int main(int argc, char *argv[])
{	
	//checking if valid number of arguments have been passed
	if(argc != 3)
	{
		printf("usage -> <./programName> <Number of unknowns> <maximum iterations>\n");
		exit(-1);
	}
	
	int Size = atoi(argv[1]);	//obtaining dimension of co-eff matrix from the argument passed by user
	int maxItr = atoi(argv[2]);	//the maximum number of iterations the program is allowed to run
	float  diffX = 0;	 	//difference between two result vectors of successive iterations
	double diag;			//diagonal element of coefficient matrix
	int iteration = 0; 	 	//iteration count
	int i;
	struct timeval tv;
	double timing;			//time taken for execution
	
	//host vectors
	float* coeffMat;         	//coeff matrix	
	float* rhsVect;          	//RHS vector
	float* solutionVect;    	//result vector
	float* diagOfCoeffMat;   	//diagonal elements of coeffMat
	float* tempSolution;	 	//new values calculated for result vector in the subsequent iteration
	
	//device vectors
	float* devCoeffMat;
	float* devRhsVect;
	float* devSolutionVect;
	float* devDiagOfCoeffMat;
	float* devTempResult; 
			               
	//generating and initalizing the required vectors in the host
	InitializeVectors(&coeffMat, &solutionVect, &rhsVect, &diagOfCoeffMat, &tempSolution, Size, &diag);
	
	//allocation of device memory
  	CUDA_SAFE_CALL(cudaMalloc( (void**)&devSolutionVect, Size * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc( (void**)&devCoeffMat, Size * Size * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc( (void**)&devRhsVect, Size * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc( (void**)&devDiagOfCoeffMat, Size * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc( (void**)&devTempResult, Size * sizeof(float)));

	//copying host to device 
	CUDA_SAFE_CALL(cudaMemcpy((void*)devCoeffMat, (void*)coeffMat, Size*Size*sizeof(float) , cudaMemcpyHostToDevice ));
	CUDA_SAFE_CALL(cudaMemcpy((void*)devRhsVect, (void*)rhsVect, Size*sizeof(float) , cudaMemcpyHostToDevice ));
	CUDA_SAFE_CALL(cudaMemcpy((void*)devSolutionVect, (void*)solutionVect, Size*sizeof(float) , cudaMemcpyHostToDevice ));
	CUDA_SAFE_CALL(cudaMemcpy((void*)devDiagOfCoeffMat, (void*)diagOfCoeffMat, Size*sizeof(float) , cudaMemcpyHostToDevice ));
	
	//defining thread grid and block
	dim3 dimGrid(1,1);
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE); 
	
	//intializing cublas
	cublasInit();
	
	//recording start time
	gettimeofday(&tv, NULL);
   	double t1=tv.tv_sec+(tv.tv_usec/1000000.0);
	
	//performing jacobian method to obtain solution parallely by calling the device kernels
   	do
	{
		//incrementing the iteration count
		iteration++;
				
		//multiplying coeffMat and solutionVect and subtracting from RHS vector (b-Ax)
		cublasScopy(Size, devRhsVect, 1, devTempResult, 1);

	        cublasSgemv ('N', Size, Size, -1, devCoeffMat, Size, devSolutionVect, 1, 1, devTempResult, 1);

		cublasScopy(Size, devTempResult, 1, devSolutionVect, 1);

		//dividing the result of the vector-vector subtraction by the diagonal elements
		ParallelScalarDiv<<<dimGrid, dimBlock>>>( devTempResult, devDiagOfCoeffMat, devSolutionVect, Size, BLOCKSIZE);
		
		//copying the result from the device
		CUDA_SAFE_CALL(cudaMemcpy((void*)tempSolution, (void*)devSolutionVect, Size * sizeof(float), cudaMemcpyDeviceToHost ));		
		
		//calculating difference in solutionVect in successive iterations (eucledian distance)
		diffX = calcDiff(solutionVect, tempSolution, Size);
		
		//assigning tempSolution to solutionVect
		for(i=0; i<Size; i++)
		{
			solutionVect[i] = tempSolution[i];
		}

	}while(diffX > ERR && iteration < maxItr);

	//recording stop time
	gettimeofday(&tv,NULL);
   	double t2=tv.tv_sec+(tv.tv_usec/1000000.0);
	
	//calculating time taken for computation
	timing = t2 - t1;	

	cublasShutdown();
	
 	//outputting the solution to both stdout and a file
	outputSolution(solutionVect, iteration, timing, Size);
	
	//freeing the memory allocated to the diffrerent vectors on host
	freeHostMemory(coeffMat, solutionVect, rhsVect, diagOfCoeffMat, tempSolution);
	
	//freeing the memory allocated to the diffrerent vectors on device
	cudaFree(devSolutionVect);
	cudaFree(devCoeffMat);
	cudaFree(devRhsVect);
	cudaFree(devDiagOfCoeffMat);
	cudaFree(devTempResult);
	
	return 0;	
}
//---------------------------------------------------------------------------------------------------------------------------
void InitializeVectors(float** coeffMat, float** solutionVect, float** rhsVect, float** diagOfCoeffMat, float** tempSolution, int Size, double *diag)
{
	generateCoeffMat(Size, coeffMat, diag);
	generateRhsVect(Size, rhsVect, *coeffMat);
	generateSolutionVect(Size, solutionVect);
	generateDiagOfCoeffMat(Size, diagOfCoeffMat, *coeffMat);
	generateTempSolution(tempSolution,Size);
	return;
}
//---------------------------------------------------------------------------------------------------------------------------
void   generateCoeffMat(int Size, float** coeffMat, double *diag)
{
	int i,j;
	*diag = 0;  //sum of the row element of the coeff matrix
	
	//assigining and verifying availability of memory	
	*coeffMat = (float *)malloc_safe_call(Size*Size*sizeof(float));	
	
	//assigning values to the coefficient matrix in column major order
	for(j=0; j<Size; j++)
		for(i=0; i<Size; i++)
			(*coeffMat)[i+j*Size] = j+1;

	//finding sum of row elements and making it diagonal element
	for(i=0; i<Size; i++)
		(*diag) += (*coeffMat)[i*Size];

	(*diag) = (*diag) * DOMINANCE;

	//making the matrix diagonal dominant
	for(i=0; i<Size; i++)
		(*coeffMat)[i*Size + i] = (*diag);
	return ;
}
//---------------------------------------------------------------------------------------------------------------------------
void   generateRhsVect(int Size, float** rhsVect, float* coeffMat)
{
	int i,j;
		
	//assigining and verifying availability of memory	
	*rhsVect = (float *)malloc_safe_call(Size*sizeof(float));
	
	//initializing rhsVect to 0's
	for(i=0; i<Size; i++)
		(*rhsVect)[i] = 0;
	//assigning values to rhsVect 
	for(i=0; i<Size; i++)
		for(j=0; j<Size; j++)
			(*rhsVect)[i] += coeffMat[j*Size + i];
	return;
}
//---------------------------------------------------------------------------------------------------------------------------
void   generateSolutionVect(int Size, float** solutionVect)
{
	int i;
	
	//assigining and verifying availability of memory
	*solutionVect = (float *)malloc_safe_call(Size*sizeof(float));
	
	//initial solution for x vector
	for(i=0; i<Size; i++)
		(*solutionVect)[i] = 0;
	return;
}
//---------------------------------------------------------------------------------------------------------------------------
void   generateDiagOfCoeffMat(int Size, float** diagOfCoeffMat, float* coeffMat)
{
	int i=0;
	
	//assigining and verifying availability of memory
	*diagOfCoeffMat = (float *)malloc_safe_call(Size*sizeof(float));
	
	//values of diagOfCoeffMat
	for(i=0; i<Size; i++)
		(*diagOfCoeffMat)[i] = coeffMat[i*Size+i];
	//forcing diag elements of coeffMat matrix to zero for purpose of calculation
	for(i=0; i<Size; i++)
		coeffMat[i*Size+i] = 0;
	return;
}
//---------------------------------------------------------------------------------------------------------------------------
void   generateTempSolution(float** tempSolution,int Size)
{
	//assigining and verifying availability of memory
	*tempSolution = (float *)malloc_safe_call(Size*sizeof(float));
	
	return;
}
//--------------------------------------------------------------------------------------------------------------------------
float  calcDiff(float* solutionVect, float* tempSolution, int Size)
{
	int i=0;
	float diffX = 0;
	for(i=0; i<Size; i++)
		diffX += pow((tempSolution[i] - solutionVect[i]), 2);
	diffX = sqrt(diffX);
	return(diffX);
}
//--------------------------------------------------------------------------------------------------------------------------
void   outputSolution(float* solutionVect, int iteration, double timing, int Size)
{
	int i=0;
	FILE *fp;	//file pointer
	
	//opening file to write solution
	fp = fopen("gpuCublasSolution.txt","w+");
	if(fp == NULL)
	{
		printf("Can't open the file gpuCublasSolution\n");
		exit(-1);
	}
	
	//displaying final result
	for(i=0; i<Size; i++)
	{
		printf("%f\n", solutionVect[i]);	//printing to stdout
		fprintf(fp, "%f\n", solutionVect[i]);   //printing to file
	}	
	
	printf("\nx vector displayed above is calculated in %d iterations and in %lf secs\n\n", iteration, timing);
	fclose(fp);
		
}
//--------------------------------------------------------------------------------------------------------------------------
void   freeHostMemory(float* coeffMat, float* solutionVect, float* rhsVect, float* diagOfCoeffMat, float* tempSolution)
{
	free(coeffMat);
	free(solutionVect);
	free(rhsVect);
	free(diagOfCoeffMat);
	free(tempSolution);
	
	return;
}
//--------------------------------------------------------------------------------------------------------------------------
void *malloc_safe_call(int size)
{
	void *ptr;
	
	ptr = malloc(size);
	
	if(ptr==NULL)
	{
		printf("memory unavailable\n");
		exit(-1);
	}	
	
	return(ptr);	
}
//--------------------------------------------------------------------------------------------------------------------------
