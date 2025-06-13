/*********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                       October 15-18, 2013

  Example     : SPmv_cudpp.cu

  Objective   : CUDA code using CUDPP library for sparse_sp_matrix vector 
                multiplication using Compressed Sparse Row(CSR) format

  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

****************************************************************/
// .................... CDAC Modified the code ........................ 

//CUDA code using CUDPP library for sparse_sp_matrix vector multiplication using Compressed Sparse Row(CSR) format
//This code is a modified version of the code obtained at http://www.nada.kth.se/~tomaso/gpu08/sptest.cu


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cudpp.h"

double gettime();
void sprand(int nrows,int ncols,int nnz,int row_ptr[], int col_idx[],float matrix[]);
void*  malloc_safe_call(int size);

/************************************************************************************
*  pragma routine to report the detail of cuda error
************************************************************************************/
#define CUDA_SAFE_CALL(call)                                                    \
            do{                                                                 \
                cudaError_t err = call;                                         \
                if(err != cudaSuccess)                                          \
                 {                                                              \
                   fprintf(stderr, "Cuda error in file '%s' in line %i\n",	\
                   __FILE__, __LINE__ );              				\
                   exit(1);                                                     \
                 }                                                              \
             } while (0)                                                        \


__global__ static void zero(int n,float v[])
{
	const int pid = threadIdx.x+blockIdx.x*blockDim.x;
	const int  np = blockDim.x*gridDim.x;
	int i;
	for(i = pid; i<n; i+=np) v[i] = 0.0;
}


int main(int argc, char* argv[])
{
	if(argc != 3)
	{
		printf("USAGE: <./program_name> <./no. of rows> <sparsity in the range of 0 to 1>\n");
		exit(-1);
	}

	int n = atoi(argv[1]);		//size of vector
	float sparsity = atof(argv[2]); //percentage of zero's
	int nnz = n*n*(1 - sparsity);	//number of non-zeros's
	int *row_ptr;			//row vector of the sparse matrix
	int *col_idx;			//column index vector of the sparse matrix
	float *matrix,*vector, *result, *dev_vector,*dev_result;
	int i;
	
	row_ptr = (int *)malloc_safe_call(sizeof(int) * (n+1));
	col_idx   = (int *)malloc_safe_call(sizeof(int) * nnz);
	matrix 	  = (float *)malloc_safe_call(sizeof(float) * nnz);

	//generate the sparse martrix and row_ptr and col_idx vectors
	sprand(n,n,nnz,row_ptr,col_idx,matrix);

	vector  = (float *)malloc_safe_call(sizeof(float) * n);
	result =  (float *)malloc_safe_call(sizeof(float) * n);
	
	//assign values to the vector used in multiplication
	for(i = 0; i<n; i++)
	{
		vector[i] = (rand() / (float) RAND_MAX > 0.5);
	}

	//create configuration
	CUDPPConfiguration config;
	config.datatype = CUDPP_FLOAT;
	config.options = (CUDPPOption)0;
	config.algorithm = CUDPP_SPMVMULT;

	//create sparse matrix object
	CUDPPHandle sparseMatrixHandle;
	CUDPPResult cudpp_result = CUDPP_SUCCESS;
	cudpp_result = cudppSparseMatrix(&sparseMatrixHandle, config, nnz, n, (void *) matrix, (unsigned int *) row_ptr, 
				    (unsigned int *)col_idx);

	if (cudpp_result != CUDPP_SUCCESS)
	{
		fprintf(stderr, "Error creating Sparse matrix object\n");
		return 1;
	}

	CUDA_SAFE_CALL(cudaMalloc((void **) &dev_vector,sizeof(float)*n));
	CUDA_SAFE_CALL(cudaMalloc((void **) &dev_result,sizeof(float)*n));
	CUDA_SAFE_CALL(cudaMemcpy(dev_vector,vector,sizeof(float)*n,cudaMemcpyHostToDevice));

	//start timing computation
	double t0 = gettime();
	
	//intialize dev_result vector
	zero<<<14*6,128>>>(n,dev_result);
	
	//call sparse_matrix vector multiplication cudpp library function
	cudppSparseMatrixVectorMultiply(sparseMatrixHandle, dev_result, dev_vector);

	cudaThreadSynchronize();
	
	//stop timing computation
	double t1 = gettime();
	printf("time taken for computation is %lf seconds\n", t1-t0);

	CUDA_SAFE_CALL(cudaMemcpy(result,dev_result,sizeof(float)*n,cudaMemcpyDeviceToHost));
	
	cudaFree(dev_result);
	cudaFree(dev_vector);
	
	free(result);
	free(vector);
	free(matrix);
	free(col_idx);
	free(row_ptr);

	return 0;
}


void sprand(int nrows,int ncols,int nnz,int row_ptr[], int col_idx[],float matrix[])
{
	int i,j,k,n;
	double r;

	n = nrows * ncols;
	k = 0;

	for(i = 0; i < nrows; i++)
	{
		row_ptr[i] = k;
		
		for(j = 0; j < ncols; j++)
		{
			r = rand() / (double) RAND_MAX;
			
			if(r * (n - (i * ncols + j)) < (nnz - k))
			{
				col_idx[k] = j;
				matrix[k] = 1.0;
				k = k+1;
			}
		}
	}

	row_ptr[nrows] = k;
}

double gettime()
{
	struct timeval tv;
	gettimeofday(&tv,NULL);
	return tv.tv_sec + 1e-6*tv.tv_usec;
}

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
