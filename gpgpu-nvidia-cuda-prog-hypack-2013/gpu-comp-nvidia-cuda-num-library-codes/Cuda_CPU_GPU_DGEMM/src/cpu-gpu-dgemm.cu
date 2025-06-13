/*********************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

Objective : To carry matrix matrix multiplication by partition the Matrix B 
            between CPU+GPU using CBLAS+CUBLAS - BLAS3 library functions	 
Input	  : Size of the matrix row size, matrix column size 
Output	  : Time Taken for computation , Gflop/s 

Created   : August-2013

E-mail    : hpcfte@cdac.in     

********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
# include <sys/time.h>

/*CUBLAS library initialization */
#include <cuda.h>
#include "cublas.h"
#include <cuda_runtime.h>

/* CBLAS header file inclusion*/
extern "C" 
{
	#include <mkl.h>
//	#include <cblas.h>;
}


#define EPS 1.0e-14 /* threshhold aprrox epsilion value */ 

#define TOTALEVENT 5  /* Toltal number of GPU event to be recorded */

#define LINE_DOT "\n..................................................................................\n"
#define LINE "\n_________________________________________________________________________________________\n"

/* Function declaration of function which checks GPU results against CPU results */
void  checkResult(double *InMatA, double *InMatB, double *outMatC, int m,int n , int k);

/* Main function */
int main(int argc, char** argv)
{
	int 		rowA, rowB, rowC, colA, colB, colC; /* holds matrices dimensions */
	int		M, K, N, i, n_cpu, n_gpu, lda, ldb, ldc; /* holds matrices dimension and leading dimension */
	double		alpha = 1.0, beta = 0.0; 
	double*		hMatA;	/* host matrix A */
	double*		hMatB;  /* host matrix B */
	double*		hMatC;  /* host matrix C */
	double*		dMatA;	/* device matrix A */
	double*		dMatB;  /* device matrix B */
	double*		dMatC;  /* device output matrix C */
	float 		*elapsedTime; /* holds total elapsed time */
	double  	mflops, gflops;	/* holds FLOPS */
	double 		Tsec_cpu=0.0,Tsec_gpu=0.0; /* holds time taken for computation on CPU, GPU */
        struct 	timeval	tv_start, tv_end; /* variable to hold start / end time for CPU*/
	cudaEvent_t  	*start, *stop;
	cublasStatus 	status;		/* holds status of function calls */
 	cudaError_t     Error;		/* holds error return from function calls */

	cudaDeviceProp  prop; /* device property */
        int             device;	/* holds number of devices */


	//printf("\n ***************** Code  : CPU + GPU Dgemm (BLAS3 + CUBLAS3) ***********************\n");

	if (argc != 4 ){
		printf("\n Invalid number of arguments : <./executable>  <rowA > <colA/rowsB> <colB> \n");
		exit(-1);
	}	
	
	/********************************************
         Device Properties capturing
	*********************************************/
        Error = cudaGetDevice (&device);
        if (Error != cudaSuccess){
        printf ("%s", Error);
	exit (-1);
	}

        Error = cudaGetDeviceProperties (&prop, device);
        if (Error != cudaSuccess){
        printf ("%s", Error);
	exit (-1);
	}
        else{
      // printf ("\n\t Device name: %s", prop.name);
      // printf ("\n\t Majornumber: %d", prop.major);
      // printf ("\n\t Minornumber: %d", prop.minor);
        }

	/**************************************************
         Verifying Compute Capabilities >= 1.3
	***************************************************/
        if ((prop.major >= 1 && prop.minor >= 3) || (prop.major >= 2 && prop.minor >= 0)){
      // printf ("\n\t Device is capable of double precision computation !! \n");
	}
        else{
        printf ("\n\t Device is NOT capable of double precision computation !! \n Exiting ... \n");
        exit (-1);
        }

	 /* read matrices size from command line arguments */
	 M = atoi(argv[1]) ;
         K = atoi(argv[2]);
         N = atoi(argv[3]);

         
	if ( N % 2 !=0 ) {
	 	 printf("\n  Usage : <./executable> <rowA > <colA/rowsB> <colB> ");
               	 printf(" \n Column of Matix B [ argv[3] ] should be divisible by 2 .. exiting \n\n");
                exit(-1); 


	}
	
	rowA = M ; /* set rows of matrix A */
  	colA = rowB = K ; /* set column of matrix A and rows of matrix B */
        colB = N ; /* set column of matrix B*/
		
	rowC = M ; /* set rows of resultant matrix C */
	colC = N ; /* set columns of resultant matrix C */


	/* Division of Data between CPU / GPU */
	n_gpu = (N - (N/2));
        n_cpu = (N - n_gpu); 

        lda = M ; ldb = K ; ldc = M; /* set the leading dimension of matrices */


	/* allocate memory for GPU events */
	start = (cudaEvent_t*) malloc (sizeof(cudaEvent_t)* TOTALEVENT );
	stop = (cudaEvent_t*) malloc (sizeof(cudaEvent_t) * TOTALEVENT);
	elapsedTime = (float*) malloc (sizeof(float) * TOTALEVENT);
	

	/* CUBlas in itialization */
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS){
		printf("\n CUBLAS initialization error !!\n");
		exit (0);
	}

 	/* Memory Allocation of the Matrices on the Host using Pinned Memory */
        Error = cudaMallocHost ((void **)&hMatA, sizeof(double) * rowA * colA);
        if (Error != cudaSuccess){
        printf ("\n Error in Host Memory Allocation - Matrix A \n");
        exit (-1);
        }

        Error = cudaMallocHost ((void **)&hMatB, sizeof(double) * rowB * colB);
        if (Error != cudaSuccess){
        printf ("\n Error in Host Memory Allocation - Matrix B \n");
        exit (-1);
        }

        Error = cudaMallocHost ((void **)&hMatC, sizeof(double) * rowC * colC);
        if (Error != cudaSuccess){
        printf ("\n Error in Host Memory Allocation - Matrix C \n");
        exit (-1);
	}

	/*************************************************************************
         * Un-comment the below section for memory allocation using Pageable Memory
	*************************************************************************/

/*

 	// Memory Allocation of the Matrices on the Host using pageable  Memory 
 	 hMatA = (double*) malloc (rowA * colA * sizeof(double));
        if (hMatA == 0){
        printf("\n Memory allocation Failed for Matrix A");
        exit (0);
        }

        hMatB = (double*) malloc (rowB * colB * sizeof(double));
        if (hMatB == 0){
        printf("\n Memory allocation Failed for Matrix B");
        exit (0);
        }

        hMatC = (double*) malloc (rowC * colC * sizeof(double));
        if (hMatC == 0){
        printf("\n Memory allocation Failed for Resultant Matrix");
        exit (0);
        }

*/
	/* Filling the Matrix with Values*/
	for (i = 0; i < rowA * colA; i++){
		hMatA[i] = drand48();
	}
	for (i=0; i < rowB * colB; i++){
		hMatB[i] = drand48(); 
	}
	for (i=0; i < rowC * colC; i++){
		hMatC[i] = 0.0 ; 
	}

 
	/* Allocating memory for Matrices on Device*/
	status = cublasAlloc (rowA * colA, sizeof(double), (void**)&dMatA);
	if (status != CUBLAS_STATUS_SUCCESS){
		printf("\n CUBLAS Memory Allocation error for dMatA!! \n");
		exit (0);
	}
	
	status = cublasAlloc (rowB * n_gpu, sizeof(double), (void**)&dMatB);
	if (status != CUBLAS_STATUS_SUCCESS){
		printf("\n CUBLAS Memory Allocation error dMatB!! \n");
		exit (0);
	}
	status = cublasAlloc (rowC * n_gpu , sizeof(double), (void**)&dMatC);
	if (status != CUBLAS_STATUS_SUCCESS){
		printf("\n CUBLAS Memory Allocation error for dMatC!! \n");
		exit (0);
	}

	for ( i = 0; i<TOTALEVENT ; i++) {
	
		/* Creating the Events */
        	cudaEventCreate (&start[i]);
        	cudaEventCreate (&stop[i]);
	}
	
	cudaEventRecord (start[0], 0);
	/* Initialization of device matrix A with host matrix A data */
	status = cublasSetMatrix (rowA, colA, sizeof(double), hMatA, lda , dMatA, lda);
	if (status != CUBLAS_STATUS_SUCCESS){
		printf("\n CUBLAS Value Initialization error of dMatA!! \n");
		exit (0);
	}
	cudaEventRecord (stop[0], 0);
  	cudaEventSynchronize (stop[0]);	
	
	cudaEventRecord (start[1], 0);
	/* Initialization of device matrix B  with host matrix B data */
	status = cublasSetMatrix (rowB, n_gpu, sizeof(double),hMatB, ldb, dMatB, ldb);
	if (status != CUBLAS_STATUS_SUCCESS){
		printf("\n CUBLAS Value Initialization error of matrix dMatB!! \n");
		exit (0);
	}
	cudaEventRecord (stop[1], 0);
  	cudaEventSynchronize (stop[1]);	


	cudaEventRecord (start[2], 0);
	status = cublasSetMatrix (rowC, n_gpu, sizeof(double),hMatC, ldc, dMatC, ldc);
	if (status != CUBLAS_STATUS_SUCCESS){
		printf("\n CUBLAS Value Initialization error of matrix dMatC!! \n");
		exit (0);
	}
	cudaEventRecord (stop[2], 0);
  	cudaEventSynchronize (stop[2]);	

	cudaEventRecord (start[3], 0);
	/*** Performs operation using CUBLAS DGEMM ******/
	cublasDgemm ('N', 'N', M , n_gpu , K  , alpha, dMatA, lda, dMatB, ldb , beta, dMatC, ldc );
	cudaEventRecord (stop[3], 0);
  	cudaEventSynchronize (stop[3]);	

	
	gettimeofday(&tv_start,0);
	/***** Performs operation using CBLAS DGEMM*******/
	cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans, M, n_cpu , K, alpha, hMatA, lda , hMatB+K*n_gpu , ldb, beta, hMatC+M*n_gpu, ldc);
	gettimeofday(&tv_end,0);

	
	 cudaEventRecord (start[4], 0);	
	/**** copy output matrix from device to host */
	status = cublasGetMatrix (M, n_gpu, sizeof(double), dMatC, ldc, hMatC, ldc);
        if (status != CUBLAS_STATUS_SUCCESS){
        printf("\n CUBLAS Error in reading the result back !! \n");
        exit (0);
	}
	 cudaEventRecord (stop[4], 0);
  	cudaEventSynchronize (stop[4]);	
	

	/* compute elapsed time for each operation */
	for ( i=0 ; i< TOTALEVENT ; i++) {
		cudaEventElapsedTime ( &elapsedTime[i], start[i], stop[i]);
	}

	/* compute total time taken for computation on CPU */
	Tsec_cpu = ((double)tv_end.tv_sec + (((double) (tv_end.tv_usec) /1000000.0))) - ((double)tv_start.tv_sec + (((double) (tv_start.tv_usec) /1000000.0)))  ;
	
	/* Compute total computation on GPU */
	for ( i=0 ; i< TOTALEVENT ; i++) {
		Tsec_gpu += (double) (elapsedTime[i]  * 1.0e-3);
	}


	/* compute FLOPS */
	mflops= ((1.0e-6 * ((2. * M *K * n_cpu) /Tsec_cpu)) + (1.0e-6 * (( 2.0 * M * K * n_gpu)/Tsec_gpu)));
	gflops= (1.0e-3 * mflops );


        printf(LINE_DOT);
        printf("Matrix-Size \t Comp Time(CPU+GPU Sec) \t  CBLAS GFlops ");
        printf(LINE_DOT);
        printf("\n %d * %d  \t %.8lf  \t\t\t %.8lf", M, N,(Tsec_cpu + Tsec_gpu)  , gflops );
        printf(LINE_DOT);


	/* check CPU+GPU results against CPU results */
        checkResult( hMatA, hMatB, hMatC,M,N,K);
        printf(LINE);
        printf("\n");

/*
	printf("\n Printing resultant Matrix \n");
	for (i=0; i < M * N; i++){
		printf("\t %f \n",hMatC[i]);
	}

*/


	/* Destroy cuda event */
	 for ( i=0 ; i< TOTALEVENT ; i++) {
                cudaEventDestroy(start[i]);
                cudaEventDestroy(stop[i]);
        }




	 /* Free the memory on Host */
        Error = cudaFreeHost(hMatA);
        if (Error != cudaSuccess){
        printf("\n Error in freeing the memory of matrix A on Host !! \n ");
        exit (0);
        }

        Error = cudaFreeHost(hMatB);
        if (Error != cudaSuccess){
        printf("\n Error in freeing the memory of matrix B on Host !! \n");
        exit (0);
        }

        Error = cudaFreeHost(hMatC);
        if (Error != cudaSuccess){
        printf("\n Error in freeing the memory of matrix C on Host !! \n");
        exit (0); 
        }


	/* Free the memory on Device */
	status = cublasFree(dMatA);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Error in freeing the memory of matrix A on Device !! \n");
	exit (0);
	}
	
	status = cublasFree(dMatB);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Error in freeing the memory of matrix B on Device !! \n");
	exit (0);
	}
	
	status = cublasFree(dMatC);
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Error in freeing the memory of matrix C on Device !! \n");
	exit (0);
	}


	/* Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS){
	printf("\n CUBLAS Shutdwon Error !! \n");
	exit (0);
	}

	free(start);
        free(stop);
        free(elapsedTime);

	 /********************************
         Un-comment the below section if memory is allocated using pageable memory
        **********************************/


/*   
	 free(hMatA);
        free(hMatB);
        free(hMatC);

*/

}

/***********************************************************************************
Routine for verifiying the CPU+GPU results against the CPU results
************************************************************************************/
void checkResult(double *InMatA, double *InMatB, double *outMatC, int m, int n , int k )
{
        double  alpha = 1.0, beta = 0.0;
        int     i;
        //int     j;
        //int     k1;
        double  *tempOut;
        double  errorNorm = 0.0;
        double  eps=EPS;
        double  relativeError=0.0;
	int 	flag=0;

        tempOut  = (double*) malloc (m * n * sizeof(double));
        if (tempOut == 0){
        printf("\n Memory allocation Failed for Resultant Matrix");
        exit (0);
        }

	/* CPU Compuation Performs operation using CBLAS */
        cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans, m, n , k, alpha, InMatA, m , InMatB , k, beta, tempOut, m);

        /****************************************************************** 
	Serial computation
	uncomment the below section if want to do the CPU computation
	using i,j,k loop method. Method work only for square matrices.
	 *******************************************************************/
       /*  for (i = 0; i < n  ; ++i) {
                for (j = 0; j < n; ++j) {
                        double  cprod = 0;
                        for (k1 = 0; k1 < n; ++k1) {
                                cprod += InMatA[k1 * n + i] * InMatB[j * n + k1];
                        }
                tempOut[j * n + i] = alpha * cprod + beta * tempOut[j * n + i];

                }
        }
	*/

        /*** check relative error with approx precision ****/
        for( i = 0; i < m*n; ++i) {

                if (fabs(tempOut[i]) > fabs(outMatC[i]))
                        relativeError = fabs((tempOut[i] - outMatC[i]) / tempOut[i]);
                else
                        relativeError = fabs((outMatC[i] - tempOut[i]) / outMatC[i]);

                if (relativeError > eps && relativeError != 0.0e+00 ){
                        if(errorNorm < relativeError) {
                        errorNorm = relativeError;
			flag=1;
                        }
                }

        }
	if( flag == 1) {

		//printf(" \n Results verfication : Failed");
		printf(" \n Considered machine precision : %e", eps);
		printf(" \n Relative Error 		    : %e", errorNorm);	

	}
	else {

		printf("\n Results verfication : Success");
	}

	free(tempOut);
}

