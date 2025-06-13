/*
***********************************************************************
		C-DAC Tech Workshop : hyPACK-2013
                        October 15-18, 2013

   Example 1	     :  matrix-matrix-multiply-dgemm-mkl-native.c

   Objective         :  To implement Matrix Matrix multiplication 
                        Algorithm using openMP on Xeon Phi Coprocessor

   Input             :  Automatic input generation  of Input Matrix data 
                        Size of the Square Matrix 

   Output            :  Print the Gflop/s and output Matrix C 
                        Time Elapsed and GFLOPS

   Created           :  August-2013

   E-mail            :  hpcfte@cdac.in     

************************************************************************
*/

#include <stdlib.h>
#include <stdio.h>

#include "omp.h"

#pragma native_attribute(push, target(mic))
#include "mkl.h"
#pragma native_attribute(pop)
#pragma UNROLL_AND_JAM


//#define THREADS 168
#define  NITERS 3


void local_dgemm(int N, int LD, double *A, double *B, double *C)
{
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
			N, N, N, 1.0, A, LD, B, LD, 1.0, C, LD);
}

double native_dgemm(int N, int LD, double *A, double *B, double *C)
{
	double t;
	static int first_run = 1;

	t = dsecnd();


	local_dgemm(N, LD, A, B, C);


	t = dsecnd() - t;

	first_run = 0;
	return t;
}


void bench_dgemm(int use_native, int N)
{
	/* Choose such leading dimension that there is no cache aliasing. */
	int LD = (N % 512) ? N : N + 128;

	/* Allocate memory using MKL function to make sure the addresses are
 * 	 * properly aligned. */
//	double *A = mkl_malloc(sizeof(double) * N * LD, 4096);
//	double *B = mkl_malloc(sizeof(double) * N * LD, 4096);
//	double *C = mkl_malloc(sizeof(double) * N * LD, 4096);

	double *Matrix_A = mkl_malloc(sizeof(double) * N * LD, 64);
	double *Matrix_B = mkl_malloc(sizeof(double) * N * LD, 64);
	double *Matrix_C = mkl_malloc(sizeof(double) * N * LD, 64);
	
	/*Initialise Matrices  */

	for(int i=0;i<N*LD;i++)
	{
		Matrix_A[i]=1.0F;
		Matrix_B[i]=2.0F;
		Matrix_C[i]=0.0F;
	}
	/* Select DGEMM : native . */
	double (*dgemm_func)(int, int, double *, double *, double *);
	dgemm_func = native_dgemm;

#pragma omp barrier

	double t = 0.0;
	for (int i = 0; i < NITERS + 1 ; i++) {
		double t_tmp = dgemm_func(N, LD, Matrix_A, Matrix_B, Matrix_C);
		/* Discard performance obtained on the warmup iteration. */
		if (i > 0) t += t_tmp;
	}

	mkl_free(Matrix_A);
	mkl_free(Matrix_B);
	mkl_free(Matrix_C);

	const double NOPS = 2.0 * N * N * N;
	double gflops = NOPS / (t * 1E9 / NITERS);
	printf("Native %dx%d DGEMM: %8.2f GFlops\n",
			N, N, gflops);
}

int main(int argc, char **argv)
{

	int N;
	if(argc<2)
	{
		printf("Syntax %s <Matrix Size>\n",argv[0]);
		exit(1);
	}
	N=atoi(argv[1]);
	printf("Matrix : %d", N);
	printf("  ITR : %d\n", NITERS);

	
	/* call fcuntion to tun MKL DGEMM on MIC */
	bench_dgemm(1, N);
	
	printf("\n-----------------------------------------------------\n\n");
	return 0;
}

