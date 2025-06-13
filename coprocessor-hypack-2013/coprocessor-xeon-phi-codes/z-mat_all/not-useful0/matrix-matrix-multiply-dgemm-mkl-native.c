/***************************************************************************************************
 *
 * 			CDAC Tech Workshop : hyPACK-2013
 *				Oct 15 - 18, 2013
 * FILE		: matrix-matrix-multiply-dgemm-mkl-native.c
 *
 * INPUT	: #Matrix Size
 *
 * OUTPUT	: Time Elapsed and GFLOPS
 *
 * EMAIL	: hpcfteg@cdac.in
 *
 **************************************************************************************************/

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
	/* The following settings will make MKL use OpenMP even when called
 	 * from an OpenMP region. */

	/* Enables Intel MKL to dynamically change the number of threads */
	mkl_set_dynamic(0);
	
	/* Enable nested parallel region */	
	omp_set_nested(1);
	
	/* Set number of MKL threads */
	mkl_set_num_threads(mkl_get_max_threads());
		
	printf("\nMKL threads = %d\n", mkl_get_max_threads());
	
	/* call fcuntion to tun MKL DGEMM on MIC */
	bench_dgemm(1, N);
	
	printf("\n-----------------------------------------------------\n\n");
	return 0;
}

