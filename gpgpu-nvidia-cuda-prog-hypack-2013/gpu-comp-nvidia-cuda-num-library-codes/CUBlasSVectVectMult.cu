
/************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

	Objective : To carry vector vector multiplication using CUBLAS - BLAS1 
                    library functions

	Input	  : Length of the vector

	Output	  : Scalar product of the two given input vectors

       Created    : August-2013

       E-mail     : hpcfte@cdac.in     

**********************************************************************/

#include <stdio.h>
#include <stdlib.h>

/*CUBLAS library initialization */
#include "cublas.h"

unsigned long long int rdtsc(void)
{
   unsigned long long int x;

   __asm__ volatile(".byte 0x0f,0x31" : "=A" (x));
   return x;
}

/* Main function */
int main(int argc, char** argv)
{
	cublasStatus status;
	
	int    veclen = 0, i = 0;
	long long int start_time, end_time;
	float* hVecA;
	float* hVecB;
	float* dVecA;
	float* dVecB;
	float  Result;
	
	printf("\n\t\t---------------------------------------------------------------------------");
	printf("\n\t\t Centre for Development of Advanced Computing (C-DAC)");
	printf("\n\t\t Email : betatest@cdac.in");
	printf("\n\t\t---------------------------------------------------------------------------");

	if (argc != 2){
	printf("\n Less number of arguments : <./executable> <veclen> \n");
	exit(0);
	}	
	
	/*input vector length */
	veclen = atoi (argv[1]);

	/* CUBlas in itialization */
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS initialization error !!\n");
	exit (0);
	}
	
	/* Allocating memory for Vector on Host*/
	hVecA = (float*) malloc (veclen * sizeof(float));
	if (hVecA == 0){
	printf("\n Memory allocation Failed for Vector A");
	exit (0);
	}

	hVecB = (float*) malloc (veclen * sizeof(float));
	if (hVecB == 0){
	printf("\n Memory allocation Failed for Vector B");
	exit (0);
	}

	/* Filling the Vectors with Values*/
	for (i = 0; i < veclen; i++){
		hVecA[i] = hVecB[i] = 2.00f;
	}

	/* Allocating memory for Vectors on Device*/
	status = cublasAlloc (veclen, sizeof(float), (void**)&dVecA);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Memory allocation error for dVecA ..!!");
	exit (0);
	}
	
	status = cublasAlloc (veclen, sizeof(float), (void**)&dVecB);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Memory allocation error for dVecB ..!!");
	exit (0);
	}

	
	/* Initialization of vectors with host vectors */
	status = cublasSetVector (veclen, sizeof(float), hVecA, 1, dVecA, 1);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS device vector initialization error dVecA ..!!");
	exit (0);
	}

	status = cublasSetVector (veclen, sizeof(float), hVecB, 1, dVecB, 1);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS device vector initialization error dVecB ..!!");
	exit (0);
	}
	
	/* Last error */
	cublasGetError();
	
	/* Performs operation using CUBLAS */
	start_time = rdtsc();
	Result = cublasSdot (veclen, dVecA, 1, dVecB, 1);
	end_time = rdtsc();
	status = cublasGetError();
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Error in performing the function ..!!");
	exit (0);
	}

	/*Printing the result */
	printf("\n\t\t Scalar Product of Two given Vectors = %f", Result);
	printf("\n\t\t Start Time = %lld", start_time);
	printf("\n\t\t End Time = %lld", end_time);
	printf("\n\t\t Time taken to calculate the product = %lld \n", end_time - start_time);

	/* Free the memory on Host */
	free(hVecA);
	free(hVecB);

	/* Free the memory on Device */
	status = cublasFree(dVecA);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Error in freeing the memory of Vector A on Device !!");
	exit (0);
	}
	
	status = cublasFree(dVecB);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Error in freeing the memory of Vector B on Device !!");
	exit (0);
	}
	
	/* Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Shutdwon Error !!");
	exit (0);
	}

}
