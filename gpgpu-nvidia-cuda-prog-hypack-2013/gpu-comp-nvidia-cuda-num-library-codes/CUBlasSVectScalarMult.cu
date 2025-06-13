
/**************************************************************

	C-DAC Tech Workshop : hyPACK-2013
                      October 15-18, 2013

     Objective : To carry vector vector multiplication using 
                    CUBLAS - BLAS1 library functions

    Input	  : Length of the vector 

    Output	  : A Resultant matrix

    Created       : August-2013

    E-mail        : hpcfte@cdac.in     

*********************************************************************/

#include <stdio.h>
#include <stdlib.h>

/*CUBLAS library initialization */
#include "cublas.h"

/* Vector Size */
#define N 10

/* Main function */
int main(int argc, char** argv)
{
	cublasStatus status;
	
	float* hVecA;
	float* hVecB;
	float* dVecA;
	float* dVecB;
	float sVal = 2.0; 
	int i = 0;

	/* CUBlas in itialization */
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS initialization error !!\n");
	return -1;
	}
	
	/* Allocating memory for Vector on Host*/
	hVecA = (float*) malloc (N * sizeof(float));
	if (hVecA == 0){
	printf("\n Memory allocation Failed for Vector A");
	return -1;
	}

	hVecB = (float*) malloc (N * sizeof(float));
	if (hVecB == 0){
	printf("\n Memory allocation Failed for Vector B");
	return -1;
	}

	/* Filling the Vectors with Values*/
	for (i = 0; i < N; i++){
		hVecA[i] = i;
		hVecB[i] = i;
	}

	/* Allocating memory for Vectors on Device*/
	status = cublasAlloc (N, sizeof(float), (void**)&dVecA);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Memory initialization error for dVecA!!");
	return -1;
	}
	
	status = cublasAlloc (N, sizeof(float), (void**)&dVecB);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Memory initialization error dVecB!!");
	return -1;
	}

	/* Initialization of vectors with host vectors */
	status = cublasSetVector (N, sizeof(float), hVecA, 1, dVecA, 1);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Memory initialization error dVecB!!");
	return -1;
	}

	status = cublasSetVector (N, sizeof(float), hVecB, 1, dVecB, 1);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Memory initialization error dVecB!!");
	return -1;
	}

	cublasGetError();
	
	/* Performs operation using CUBLAS */
	cublasSaxpy (N, sVal, dVecA, 1, dVecB, 1);
	status = cublasGetError();
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Error in performing the function!!");
	return -1;
	}

	/*Read the Result back */
	status = cublasGetVector (N, sizeof(float), dVecB, 1, hVecB, 1);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Error in reading the result back !!");
	return -1;
	}
	
	/*Printing the result */
	printf("\n Check the result...!! \n");
	for (i = 0; i < N; i++)
		printf("\t%f", hVecB[i]);
		
	/* Free the memory on Host */
	free(hVecA);
	free(hVecB);

	/* Free the memory on Device */
	status = cublasFree(dVecA);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Error in freeing the memory of Vector A on Device !!");
	return -1;
	}
	
	status = cublasFree(dVecB);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Error in freeing the memory of Vector B on Device !!");
	return -1;
	}
	
	/* Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Shutdwon Error !!");
	return -1;
	}

}
