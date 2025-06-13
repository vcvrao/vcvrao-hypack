/*
	Objective : To meausre power consumption while 
		    doing matrix matrix multiplication using CUBLAS	 
	Input	  : Size of the matrix row size, matrix column size 
	Output	  : Time Taken for computation , Gflop/s 
	Author 	  : HPC-FTEG 
*/


#include<cublas_nvml_power_kernel_functions.h>



/** 
 * print_err on consol
 * @param msg error message needs to print on consol
 * @param nodeNum node number
 * @param devNum device number
 * @param benchname benchmark name
**/
int print_error(char *msg,int nodeNum, int devNum , char *benchName)
{
        FILE *fp;
        fp = fopen(ERRORFILE, "w");
        if(fp == NULL)
        {
                printf("\n failed to open ERRORFILE file \n");
                exit(-1);
        }
        fprintf(fp, "Error: %s: %s on device :%d on node : %d", benchName, msg, devNum, nodeNum);
        fclose(fp);
        return 0;
}

/**
 * Allocate memory on host & device. 
 * transfer data from host to device then do
 * matrix multiplication using vendor supplied library (cublas_dgemm)
 * and transfer result matrix from device to host.  
 * then free all memories and reset device.
 * @param thread id 
**/
void *mat_mult(void *t)
{
  	int dataSize;
   	int GPUDevCount, GPUDevId;
 	int nodeNum = 1; 
 
	/* Set GPU device Id as 0 */
	GPUDevId = 0;
	
	/* Set Matrix Data Size */  	
	dataSize = MAT_START_SIZE;

	/* Matrix Matrix multiplication using CUBLAS LIB calls */
	int iRetVal = MatMatMult(nodeNum , GPUDevId, dataSize);
  	if (iRetVal != -1)
  	{ 
    	//	printf("\n[gpuDGEMMBenchmark] - results:\t%s\n\n", (iRetVal == 0) ? "PASSES" : "FAILED");
  	}

	sleep(25);
	sigFlag = 0;       /* send signal to watch_count thread to stop capturing power */
	sleep(3);
	
	/* Print output on the consol */
	printf(LINE_DOT);
	printf("Matrix-Size \t Comp Time(CPU+GPU Sec) \t  CBLAS GFlops ");
	printf(LINE_DOT);
	printf("\n %d * %d  \t %.8lf  \t\t\t %.8lf", dataSize, dataSize,( Tsec_gpu)  , gflops );
	printf(LINE_DOT);
	
	pthread_exit(NULL);
	
}

/*
 * This function performs matrix matrix multiplication 
 * using CUBLAS DGEMM call and check result with sequential 
 * results
*/

int MatMatMult(int nodeNum , int GPUDevId, int dataSize)
{
	int 		rowA, rowB, rowC, colA, colB, colC; /* holds matrices dimensions */
	int		M, K, N, i,  n_gpu, lda, ldb, ldc; /* holds matrices dimension and leading dimension */
	double		alpha = 1.0, beta = 0.0;
	double*		hMatA;	/* host matrix A */
	double*		hMatB;  /* host matrix B */
	double*		hMatC;  /* host matrix C */
	double*		dMatA;	/* device matrix A */
	double*		dMatB;  /* device matrix B */
	double*		dMatC;  /* device output matrix C */
	cudaEvent_t  	*start, *stop;
	cublasStatus 	status;		/* holds status of function calls */
 	cudaError_t     Error;		/* holds error return from function calls */

   	cudaSetDevice(GPUDevId);
   	//printf(" Device %d: %s\n", GPUDevId, device);

		
 	/* read matrices size from command line arguments */
 	M = dataSize ;
        K = dataSize;
        N = dataSize;

         
	if ( N % 2 !=0 ) 
	{
	 	printf("\n  Usage : <./executable> <rowA > <colA/rowsB> <colB> ");
		print_error("Col of MatB is not divisible by 2",nodeNum, GPUDevId,"GPUDGEMM Benchmark");
               	exit(-1); 
	}
	
	rowA = M ; /* set rows of matrix A */
  	colA = rowB = K ; /* set column of matrix A and rows of matrix B */
        colB = N ; /* set column of matrix B*/
		
	rowC = M ; /* set rows of resultant matrix C */
	colC = N ; /* set columns of resultant matrix C */


	n_gpu = (N );

        lda = M ; ldb = K ; ldc = M; /* set the leading dimension of matrices */


	/* allocate memory for GPU events */
	start = (cudaEvent_t*) malloc (sizeof(cudaEvent_t)* TOTALEVENT );
	if (start == NULL)
	{		
		print_error("cudaEvet memory allocation failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
		exit (-1);
	}
	stop = (cudaEvent_t*) malloc (sizeof(cudaEvent_t) * TOTALEVENT);
	if (stop == NULL)
	{		
		print_error("cudaEvet memory allocation failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
		exit (-1);
	}
	elapsedTime = (float*) malloc (sizeof(float) * TOTALEVENT);
	if (elapsedTime == NULL)
	{		
		print_error("cudaEvet memory allocation failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
		exit (-1);
	}
	


 	/* Memory Allocation of the Matrices on the Host using Pinned Memory */
        Error = cudaMallocHost ((void **)&hMatA, sizeof(double) * rowA * colA);
        if (Error != cudaSuccess)
	{
		print_error("Host memory allocation failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
        	exit (-1);
        }

        Error = cudaMallocHost ((void **)&hMatB, sizeof(double) * rowB * colB);
        if (Error != cudaSuccess)
	{
		print_error("Host memory allocation failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
        	exit (-1);
        }

        Error = cudaMallocHost ((void **)&hMatC, sizeof(double) * rowC * colC);
        if (Error != cudaSuccess)
	{
		print_error("Host memory allocation failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
        	exit (-1);
	}

	/* Filling the Matrix with Values*/
	for (i = 0; i < rowA * colA; i++)
	{
		hMatA[i] = drand48();
	}
	for (i=0; i < rowB * colB; i++)
	{
		hMatB[i] = drand48(); 
	}
	for (i=0; i < rowC * colC; i++)
	{
		hMatC[i] = 0.0 ; 
	}
	/* CUBlas in itialization */
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		print_error("CUBLAS initialization failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
		exit (0);
	}

	/* Allocating memory for Matrices on Device*/
	status = cublasAlloc (rowA * colA, sizeof(double), (void**)&dMatA);
	if (status != CUBLAS_STATUS_SUCCESS)	
	{
		print_error("CUBLAS memory allocation failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
		exit (0);
	}

	status = cublasAlloc (rowB * n_gpu, sizeof(double), (void**)&dMatB);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		print_error("CUBLAS memory allocation failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
		exit (0);
	}
	status = cublasAlloc (rowC * n_gpu , sizeof(double), (void**)&dMatC);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		print_error("CUBLAS memory allocation failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
		exit (0);
	}

	for ( i = 0; i<TOTALEVENT ; i++) 
	{
		/* Creating the Events */
        	cudaEventCreate (&start[i]);
       		cudaEventCreate (&stop[i]);
	}
	
	cudaEventRecord (start[0], 0);
	/* Initialization of device matrix A with host matrix A data */
	status = cublasSetMatrix (rowA, colA, sizeof(double), hMatA, lda , dMatA, lda);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		print_error("CUBLAS value intilization of dMatA failed",nodeNum, GPUDevId ,"GPUDGEMM Benchmark");
		exit (0);
	}
	cudaEventRecord (stop[0], 0);
  	cudaEventSynchronize (stop[0]);	
	
	cudaEventRecord (start[1], 0);
	/* Initialization of device matrix B  with host matrix B data */
	status = cublasSetMatrix (rowB, n_gpu, sizeof(double),hMatB, ldb, dMatB, ldb);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		print_error("CUBLAS value intilization of dMatB failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
		exit (0);
	}
	cudaEventRecord (stop[1], 0);
  	cudaEventSynchronize (stop[1]);	


	cudaEventRecord (start[2], 0);
	status = cublasSetMatrix (rowC, n_gpu, sizeof(double),hMatC, ldc, dMatC, ldc);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		print_error("CUBLAS value intilization of dMatC failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
		exit (0);
	}
	cudaEventRecord (stop[2], 0);
  	cudaEventSynchronize (stop[2]);	
	cudaEventRecord (start[3], 0);
	/*** Performs operation using CUBLAS DGEMM ******/
	cublasDgemm ('N', 'N', M , n_gpu , K  , alpha, dMatA, lda, dMatB, ldb , beta, dMatC, ldc );
	cudaEventRecord (stop[3], 0);
 	cudaEventSynchronize (stop[3]);	
	
	
	
	 cudaEventRecord (start[4], 0);	
	/**** copy output matrix from device to host */
	status = cublasGetMatrix (M, n_gpu, sizeof(double), dMatC, ldc, hMatC, ldc);
        if (status != CUBLAS_STATUS_SUCCESS)
	{
		print_error("CUBLAS reading result back failed",nodeNum, GPUDevId, "GPUDGEMM Benchmark");
	        exit (0);
	}
	 cudaEventRecord (stop[4], 0);
	cudaEventSynchronize (stop[4]);	

	/* compute elapsed time for each operation */
	for ( i=0 ; i< TOTALEVENT ; i++) 
	{
		cudaEventElapsedTime ( &elapsedTime[i], start[i], stop[i]);
	}
		
	/* Compute total computation on GPU */
	for ( i=0 ; i< TOTALEVENT ; i++) 
	{
		Tsec_gpu += (double) (elapsedTime[i]  * 1.0e-3);
	}


	/* compute FLOPS */
	mflops= ( (1.0e-6 * (( 2.0 * M * K * n_gpu)/Tsec_gpu)));
	gflops= (1.0e-3 * mflops );
	
	
	/* Uncomment to check CPU+GPU results against CPU results */
      	//checkResult( hMatA, hMatB, hMatC,M,N,K);	

	/* Destroy cuda event */
	for ( i=0 ; i< TOTALEVENT ; i++) 
	{
                cudaEventDestroy(start[i]);
               	cudaEventDestroy(stop[i]);
       	}
		 /* Free the memory on Host */
        Error = cudaFreeHost(hMatA);
       	if (Error != cudaSuccess)
	{
		print_error("Host Free memory of MatA failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
       		exit (0);
        }
       	 	Error = cudaFreeHost(hMatB);
        if (Error != cudaSuccess)
	{
		print_error("Host Free memory of MatB failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
       		exit (0);
        }

        Error = cudaFreeHost(hMatC);
        if (Error != cudaSuccess)
	{
		print_error("Host Free memory of MatC failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
	        exit (0); 
        }

	/* Free the memory on Device */
	status = cublasFree(dMatA);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		print_error("CUBLAS Dev Free memory of MatA failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
		exit (0);
	}

	status = cublasFree(dMatB);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		print_error("CUBLAS Dev Free memory of MatB failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
		exit (0);
	}

	status = cublasFree(dMatC);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		print_error("CUBLAS Dev Free memory of MatC failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
		exit (0);
	}
		/* Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		print_error("CUBLAS shut down failed",nodeNum, GPUDevId , "GPUDGEMM Benchmark");
		exit (0);
	}



	free(start);
       	free(stop);
        free(elapsedTime);

	cudaDeviceReset();

	 /********************************
         Un-comment the below section if memory is allocated using pageable memory
        **********************************/
	
/*   
	 free(hMatA);
        free(hMatB);
        free(hMatC);

*/
	return 0;

}


/**
 * This function compares device computed result with host computed result
 * and find out relative error between both. 
 * @param host matrix A
 * @param host matrix B
 * @param result : Bbtained from device computation
 * @param Size of matrix A
 * @param Size of matrix B
**/
int checkResult(double *InMatA, double *InMatB, double *outMatC, int m, int n , int k ) 
{
        double  alpha = 1.0, beta = 0.0;
        int     i;  
        int     j;  
        int     k1; 
        double  *tempOut;
        double  errorNorm = 0.0;
        double  eps=EPS;
        double  relativeError=0.0;
        int     flag=0;

        tempOut  = (double*) malloc (m * n * sizeof(double));
        if (tempOut == 0)
	{ 
		printf("Host allocation for check result failed\n");
		exit(-1);
        }   

        /* CPU Compuation Performs operation using CBLAS */
//        cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans, m, n , k, alpha, InMatA, m , InMatB , k, beta, tempOut, m);

        /******************************************************************
        Serial computation
        uncomment the below section if want to do the CPU computation
        using i,j,k loop method. Method work only for square matrices.
         *******************************************************************/
         for (i = 0; i < n  ; ++i) {
                for (j = 0; j < n; ++j) {
                        double  cprod = 0;
                        for (k1 = 0; k1 < n; ++k1) {
                                cprod += InMatA[k1 * n + i] * InMatB[j * n + k1];
                        }   
                tempOut[j * n + i] = alpha * cprod + beta * tempOut[j * n + i]; 
                }
        }
        

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

                printf(" \n Results verfication : Failed");
                printf(" \n Considered machine precision : %e", eps);
                printf(" \n Relative Error                  : %e", errorNorm);

        }
        else {

                printf("\n Results verfication : Success");
        }

        free(tempOut);
	return 0;
}


