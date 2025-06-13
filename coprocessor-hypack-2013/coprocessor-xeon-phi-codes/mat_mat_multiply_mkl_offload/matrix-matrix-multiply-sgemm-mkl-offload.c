/*
***********************************************************************
                C-DAC Tech Workshop : hyPACK-2013
                        October 15-18, 2013

   Example 1         :  matrix-matrix-multiply-dgemm-mkl-offload.c
  * FILE	     :  matrix-matrix-multiply-dgemm-mkl-offload.c

   Objective         :  To implement Matrix Matrix multiplication
                        Algorithm using openMP on Xeon Phi Coprocessor

   Input             :  Automatic input generation  of Input Matrix data
                        Size of the Square Matrix

   Output            :  Print the Gflop/s and output Matrix C
                        Time Elapsed and GFLOPS

   Created           :  August-2013

   E-mail            :  hpcfte@cdac.in

  *
  * EMAIL         : hpcfte@cdac.in
****************************************************************** */
 *
 
#include <stdlib.h>
#include <stdio.h>

#include "omp.h"

#pragma offload_attribute(push, target(mic))
#include "mkl.h"
#pragma offload_attribute(pop)
#pragma UNROLL_AND_JAM


//#define THREADS 168
#define  NITERS 3

int manual_sync;
omp_lock_t offload_lock;

__declspec(target(mic))
void local_dgemm(int N, int LD, double *A, double *B, double *C)
{
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
			N, N, N, 1.0, A, LD, B, LD, 1.0, C, LD);
}

double offload_dgemm(int N, int LD, double *A, double *B, double *C)
{
	double t;
	static int first_run = 1;

	t = dsecnd();

	/* If manual synchronization is enabled, set the lock. This will block if
 * 	 * MKL Automatic Offload was first to set the lock. */
	if (manual_sync)
		omp_set_lock(&offload_lock);

	/* Allocate memory on the card only on the first offload to improve
 * 	 * performance. The memory is released only when the process exits. This is
 * 	 	 * only suitable for benchmarking. */
#pragma offload target(mic:0) in(N, LD) \
		in(A: length(N*LD) alloc_if(first_run) free_if(0)) \
		in(B: length(N*LD) alloc_if(first_run) free_if(0)) \
		inout(C: length(N*LD) alloc_if(first_run) free_if(0))
		{
			local_dgemm(N, LD, A, B, C);
		}

	/* Unset the lock if manual synchronization is enabled */
	if (manual_sync)
		omp_unset_lock(&offload_lock);

	t = dsecnd() - t;

	first_run = 0;
	return t;
}

double host_ao_dgemm(int N, int LD, double *A, double *B, double *C)
{
	int card_available = 1;

	double t = dsecnd();

	if (manual_sync) {
		/* If manual synchronization is enabled, try to set the lock. If this
 * 		 * fails assume that access to coprocessor is locked and fall back to
 * 		 		 * host by temporarily disabling MKL Automatic Offload. */
		card_available = omp_test_lock(&offload_lock);
		if (card_available)
		{
			printf("\nCard avaiabale\n");
			mkl_mic_enable();
			mkl_set_num_threads(mkl_get_max_threads());
		}
		else
			mkl_mic_disable();
	}

	local_dgemm(N, LD, A, B, C);

	/* Unset the offload lock if manual synchronization is enabled. */
	if (card_available && manual_sync)
		omp_unset_lock(&offload_lock);

	return dsecnd() - t;
}

void bench_dgemm(int use_offload, int N)
{
	/* Choose such leading dimension that there is no cache aliasing. */
	int LD = (N % 512) ? N : N + 128;

	/* Allocate memory using MKL function to make sure the addresses are
 * 	 * properly aligned. */
	double *Matrix_A = mkl_malloc(sizeof(double) * N * LD, 4096);
	double *Matrix_B = mkl_malloc(sizeof(double) * N * LD, 4096);
	double *Matrix_C = mkl_malloc(sizeof(double) * N * LD, 4096);

	/*Initialization of Matrices */
	for(int i=0;i<N*LD;i++)
	{                                                 
		Matrix_A[i]=1.0F;                                      
		Matrix_B[i]=2.0F;                                                 
		Matrix_C[i]=0.0F;                                   
	}   
	/* Select DGEMM kind: offload or host/Automatic Offload. */
	double (*dgemm_func)(int, int, double *, double *, double *);
	dgemm_func = (use_offload) ? offload_dgemm : host_ao_dgemm;

#pragma omp barrier
	
	 double t = 0.0;
	for (int i = 0; i < NITERS + 1 ; i++) {
		 double t_tmp =  dgemm_func(N, LD, Matrix_A, Matrix_B, Matrix_C);
		/* Discard performance obtained on the warmup iteration. */
		if (i > 0) t += t_tmp;
	}

	mkl_free(Matrix_A);
	mkl_free(Matrix_B);
	mkl_free(Matrix_C);

	const double NOPS = 2.0 * N * N * N;
	double gflops = NOPS / (t * 1E9 / NITERS);
	printf("%s %dx%d DGEMM: %8.2f GFlops\n",
			(use_offload) ? "Offload" : "Host/AO", N, N, gflops);
}

int main(int argc, char **argv)
{
	if (argc != 4) {
		printf("Usage: %s <concurrent coprocessor access=0|1> "
				"<manual sync=0|1> <N> \n", argv[0]);
		return -1;
	}

	int concurrent = atoi(argv[1]);
	manual_sync = atoi(argv[2]);
	int N = atoi(argv[3]);
//	int THREADS = atoi(argv[4]);

	printf("Coprocessor access: %s\n", concurrent ? "concurrent" : "serial");
	printf("Manual synchronization: %s\n", manual_sync ? "on" : "off");
	printf("Matrix : %d", N);
//	printf("  Threads : %d", THREADS);
	printf("  ITR : %d\n", NITERS);
	if (concurrent) {
		/* The following settings will make MKL use OpenMP even when called
 * 		 * from an OpenMP region. */
		mkl_set_dynamic(0);
		omp_set_nested(1);
//		mkl_set_num_threads(omp_get_max_threads());
		mkl_set_num_threads(mkl_get_max_threads());
		
	}

	printf("\nMKL Threads%d\n", mkl_get_max_threads());
	if (manual_sync)
		omp_init_lock(&offload_lock);

	#pragma omp parallel for num_threads(2) if (concurrent)
	for (int i = 0; i < 2; i++)
	{
		bench_dgemm(i, N);
	}
//		bench_dgemm(1, N);
	
	printf("\n-----------------------------------------------------\n\n");
	return 0;
}

