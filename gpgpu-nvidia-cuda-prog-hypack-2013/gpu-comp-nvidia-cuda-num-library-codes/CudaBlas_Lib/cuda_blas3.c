#include <stdlib.h>
#include <stdio.h>
#include "cublas.h"
#include "cudablas.h"

/* sgemm of CUBLAS , executed on the device 
 * void
 * cublasSgemm (char transa, char transb, int m, int n, int k, float alpha,
 *              const float *A, int lda, const float *B, int ldb, float beta,
 *              float *C, int ldc)
 *
 * computes the product of matrix A and matrix B, multiplies the result
 * by a scalar alpha, and adds the sum to the product of matrix C and
 * scalar beta. sgemm() performs one of the matrix-matrix operations:
 *
 *     C = alpha * op(A) * op(B) + beta * C,
 *
 * where op(X) is one of
 *
 *     op(X) = X   or   op(X) = transpose(X)
 */
 void cudablas_sgemm(const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *host_A,
                 const int lda, const float *host_B, const int ldb,
                 const float beta, float *host_C, const int ldc)
{    
	cublasStatus status;
	float *dev_A, *dev_B, *dev_C;
	int i;
	char transA, transB;
	unsigned long long sizeA, sizeB, sizeC;
	
	if(TransA == CblasNoTrans)
		transA = 'N';
	if(TransA == CblasTrans)
		transA = 'T';
	if(TransB == CblasNoTrans)
		transB = 'N';
	if(TransB == CblasTrans)
		transB = 'T';
		
	/* Calculate the matrx size */
	sizeA = M*K;
	sizeB = K*N;
	sizeC = M*N;	

	/* Initialize CUBLAS */
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  CUBLAS initialization error.\n");
	return;}

	/* Allocate device memory for the matrices */
	status = cublasAlloc(sizeA, sizeof(float), (void**)&dev_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memeory on device (A)\n");
	return;}
	
	status = cublasAlloc(sizeB, sizeof(float), (void**)&dev_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memory on device (B) \n");
	return;}

	status = cublasAlloc(sizeC, sizeof(float), (void**)&dev_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memory on device (C)\n");
	return;}

	/* Initialize the device matrices with the host matrices */
	status = cublasSetVector(sizeA, sizeof(float), host_A, 1, dev_A, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write A on dev failed)\n");
	return;}

	status = cublasSetVector(sizeB, sizeof(float), host_B, 1, dev_B, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write B on dev failed)\n");
	return;}

	status = cublasSetVector(sizeC, sizeof(float), host_C, 1, dev_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write C on dev failed)\n");
	return;}
    
	/* Clear last error */
	cublasGetError();

	/* Performs operation (cblasSgemm) using cublas */
	
	cublasSgemm(transA, transB, M, N, K, alpha, dev_A, lda, dev_B, ldb, beta, dev_C, ldc);
	
	status = cublasGetError();
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : kernel execution error on device .\n");
	return;}

   
	/* Read the result from dev to host  */
	status = cublasGetVector(sizeC, sizeof(float), dev_C, 1, host_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (read C from dev failed)\n");
	return;}
	

	/* Memory clean up */

	status = cublasFree(dev_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Memory free error (A)\n");
	return;}

	status = cublasFree(dev_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  memory free error (B)\n");
	return;}
   
	status = cublasFree(dev_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  memory free error (C)\n");
	return;    }


	/* CUBLAS Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("\n\t Error :  CUBLAS shutdown error \n");
	return;  }
	return;
}



/* ssymm of CUBLAS , executed on the device 
 * void
 * cublasSsymm (char side, char uplo, int m, int n, float alpha,
 *              const float *A, int lda, const float *B, int ldb,
 *              float beta, float *C, int ldc);
 *
 * performs one of the matrix-matrix operations
 *
 *   C = alpha * A * B + beta * C, or
 *   C = alpha * B * A + beta * C,
 */
void cudablas_ssymm(const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const float alpha, const float *host_A, const int lda,
                 const float *host_B, const int ldb, const float beta,
                 float *host_C, const int ldc)
{    
	cublasStatus status;
	float *dev_A, *dev_B, *dev_C;
	int i;
	char side,uplo;
	unsigned long long sizeA, sizeB, sizeC;
	
	if(Uplo == CblasUpper)
		uplo = 'U';
	if(Uplo == CblasUpper)
		uplo = 'L';
	
	/* Calculate the matrx size */
	if(Side == CblasLeft)
	{
		sizeA = M*M;
		side = 'L';
	}
	if(Side == CblasRight)
	{
		sizeA = N*N;
		side = 'R';
	}
			
	sizeB = M*N;
	sizeC = M*N;	

	/* Initialize CUBLAS */
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  CUBLAS initialization error.\n");
	return ;    }

	/* Allocate device memory for the matrices */
	status = cublasAlloc(sizeA, sizeof(float), (void**)&dev_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memeory on device (A)\n");
	return ;    }
	
	status = cublasAlloc(sizeB, sizeof(float), (void**)&dev_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memory on device (B) \n");
	return ;    }

	status = cublasAlloc(sizeC, sizeof(float), (void**)&dev_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memory on device (C)\n");
	return ;    }

	/* Initialize the device matrices with the host matrices */
	status = cublasSetVector(sizeA, sizeof(float), host_A, 1, dev_A, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write A on dev failed)\n");
	return ;   }

	status = cublasSetVector(sizeB, sizeof(float), host_B, 1, dev_B, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write B on dev failed)\n");
	return ;    }

	status = cublasSetVector(sizeC, sizeof(float), host_C, 1, dev_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write C on dev failed)\n");
	return ;    }
    
	/* Clear last error */
	cublasGetError();

	/* Performs operation (cublasSymm) using cublas */
	
	cublasSsymm(side, uplo, M, N, alpha, dev_A, lda, dev_B, ldb, beta, dev_C, ldc);
	
	status = cublasGetError();
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : kernel execution error on device .\n");
	return ;    }

	    
	/* Read the result from dev to host  */
	status = cublasGetVector(sizeC, sizeof(float), dev_C, 1, host_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (read C from dev failed)\n");
	return ;    }

	/* Memory clean up */
	
	status = cublasFree(dev_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Memory free error (A)\n");
	return ;    }

	status = cublasFree(dev_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  memory free error (B)\n");
	return ;    }
   
	status = cublasFree(dev_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  memory free error (C)\n");
	return ;    }


	/* CUBLAS Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("\n\t Error :  CUBLAS shutdown error \n");
	return ;    }
	return ;
}




/* 
 * void
 * cublasSsyrk (char uplo, char trans, int n, int k, float alpha,
 *              const float *A, int lda, float beta, float *C, int ldc)
 *
 * performs one of the symmetric rank k operations
 *
 *   C = alpha * A * transpose(A) + beta * C, or
 *   C = alpha * transpose(A) * A + beta * C.
 */
void cudablas_ssyrk(const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const float alpha, const float *host_A, const int lda,
                 const float beta, float *host_C, const int ldc)
{    
	cublasStatus status;
	float *dev_A, *dev_C;
	int i;
	char trans,uplo;
	unsigned long long sizeA, sizeC;
	
	if(Uplo == CblasUpper)
		uplo = 'U';
	if(Uplo == CblasUpper)
		uplo = 'L';
		
	if(Trans == CblasNoTrans)
		trans = 'N';
	if(Trans == CblasTrans)
		trans = 'T';
		
	/* Calculate the matrx size */
	sizeC = N*N;
	sizeA = N*K;
	
	/* Initialize CUBLAS */
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  CUBLAS initialization error.\n");
	return ;    }


	/* Allocate device memory for the matrices */
	status = cublasAlloc(sizeA, sizeof(float), (void**)&dev_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memeory on device (A)\n");
	return ;    }
	
	status = cublasAlloc(sizeC, sizeof(float), (void**)&dev_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memory on device (C)\n");
	return ;    }

	/* Initialize the device matrices with the host matrices */
	status = cublasSetVector(sizeA, sizeof(float), host_A, 1, dev_A, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write A on dev failed)\n");
	return ;   }

	status = cublasSetVector(sizeC, sizeof(float), host_C, 1, dev_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write C on dev failed)\n");
	return ;    }
    
	/* Clear last error */
	cublasGetError();

	/* Performs operation (cblasSsyrk) using cublas */
	
	cublasSsyrk(uplo, trans, N, K, alpha, dev_A, lda, beta, dev_C, ldc);

	status = cublasGetError();
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : kernel execution error on device .\n");
	return ;    }

	/* Read the result from dev to host  */
	status = cublasGetVector(sizeC, sizeof(float), dev_C, 1, host_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (read C from dev failed)\n");
	return ;    }

	/* Memory clean up */
	
	status = cublasFree(dev_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Memory free error (A)\n");
	return ;    }

	status = cublasFree(dev_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  memory free error (C)\n");
	return ;    }

	/* CUBLAS Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("\n\t Error :  CUBLAS shutdown error \n");
	return ;    }
	return ;
}




/*
 * void
 * cublasSsyr2k (char uplo, char trans, int n, int k, float alpha,
 *               const float *A, int lda, const float *B, int ldb,
 *               float beta, float *C, int ldc)
 *
 * performs one of the symmetric rank 2k operations
 *
 *    C = alpha * A * transpose(B) + alpha * B * transpose(A) + beta * C, or
 *    C = alpha * transpose(A) * B + alpha * transpose(B) * A + beta * C.
 */
void cudablas_ssyr2k(const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const float alpha, const float *host_A, const int lda,
                  const float *host_B, const int ldb, const float beta,
                  float *host_C, const int ldc)
{    
	cublasStatus status;
	float *dev_A, *dev_B, *dev_C;
	int i;
	char trans, uplo;
	unsigned long long sizeA, sizeB, sizeC;
	
	if(Uplo == CblasUpper)
		uplo = 'U';
	if(Uplo == CblasUpper)
		uplo = 'L';
		
	if(Trans == CblasNoTrans)
		trans = 'N';
	if(Trans == CblasTrans)
		trans = 'T';
	
	/* Calculate the matrx size */
	sizeA = N*K;
	sizeB = N*K;
	sizeC = N*N;	

	/* Initialize CUBLAS */
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  CUBLAS initialization error.\n");
	return ;    }

	/* Allocate device memory for the matrices */
	status = cublasAlloc(sizeA, sizeof(float), (void**)&dev_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memeory on device (A)\n");
	return ;    }
	
	status = cublasAlloc(sizeB, sizeof(float), (void**)&dev_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memory on device (B) \n");
	return ;    }

	status = cublasAlloc(sizeC, sizeof(float), (void**)&dev_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memory on device (C)\n");
	return ;    }

	/* Initialize the device matrices with the host matrices */
	status = cublasSetVector(sizeA, sizeof(float), host_A, 1, dev_A, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write A on dev failed)\n");
	return ;   }

	status = cublasSetVector(sizeB, sizeof(float), host_B, 1, dev_B, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write B on dev failed)\n");
	return ;    }

	status = cublasSetVector(sizeC, sizeof(float), host_C, 1, dev_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write C on dev failed)\n");
	return ;    }
    
	/* Clear last error */
	cublasGetError();

	/* Performs operation (cblasSsyr2k) using cublas */
	
	cublasSsyr2k(uplo, trans, N, K, alpha, dev_A, lda, dev_B, ldb, beta, dev_C, ldc);
	
	status = cublasGetError();
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : kernel execution error on device .\n");
	return ;    }

	/* Read the result from dev to host  */
	status = cublasGetVector(sizeC, sizeof(float), dev_C, 1, host_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (read C from dev failed)\n");
	return ;    }

	/* Memory clean up */
	
	status = cublasFree(dev_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Memory free error (A)\n");
	return ;    }

	status = cublasFree(dev_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  memory free error (B)\n");
	return ;    }
   
	status = cublasFree(dev_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  memory free error (C)\n");
	return ;    }


	/* CUBLAS Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("\n\t Error :  CUBLAS shutdown error \n");
	return ;    }
	return ;
}


/*
 * void
 * cublasStrmm (char side, char uplo, char transa, char diag, int m, int n,
 *              float alpha, const float *A, int lda, const float *B, int ldb)
 *
 * performs one of the matrix-matrix operations
 *
 *   B = alpha * op(A) * B,  or  B = alpha * B * op(A)
 */
void cudablas_strmm(const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const float alpha, const float *host_A, const int lda,
                 float *host_B, const int ldb)
{    
	cublasStatus status;
	float *dev_A, *dev_B;
	int i;
	char side, uplo, transA, diag;
	unsigned long long sizeA, sizeB;
	
	if(Side == CblasLeft)
		side = 'L';
	if(Side == CblasRight)
		side = 'R';
		
	if(Uplo == CblasUpper)
		uplo = 'U';
	if(Uplo == CblasUpper)
		uplo = 'L';
		
	if(TransA == CblasNoTrans)
		transA = 'N';
	if(TransA == CblasTrans)
		transA = 'T';
		
	if(Diag == CblasUnit)
		diag = 'U';
	if(Diag == CblasNonUnit)
		diag = 'N';
	
	/* Calculate the matrx size */
	sizeB = M*N;
	sizeA = sizeof(host_A);	

	/* Initialize CUBLAS */
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  CUBLAS initialization error.\n");
	return ;    }

	/* Allocate device memory for the matrices */
	status = cublasAlloc(sizeA, sizeof(float), (void**)&dev_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memeory on device (A)\n");
	return ;    }
	
	status = cublasAlloc(sizeB, sizeof(float), (void**)&dev_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memory on device (B) \n");
	return ;    }

	/* Initialize the device matrices with the host matrices */
	status = cublasSetVector(sizeA, sizeof(float), host_A, 1, dev_A, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write A on dev failed)\n");
	return ;   }

	status = cublasSetVector(sizeB, sizeof(float), host_B, 1, dev_B, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write B on dev failed)\n");
	return ;    }

	/* Clear last error */
	cublasGetError();

	/* Performs operation cublasStrmm using cublas */

	cublasStrmm(side, uplo,transA, diag, M, N, alpha, dev_A, lda, dev_B, ldb);

	status = cublasGetError();
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : kernel execution error on device .\n");
	return ;    }

	/* Read the result from dev to host  */
	status = cublasGetVector(sizeB, sizeof(float), dev_B, 1, host_B, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (read C from dev failed)\n");
	return ;    }

	/* Memory clean up */
	
	status = cublasFree(dev_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Memory free error (A)\n");
	return ;    }

	status = cublasFree(dev_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  memory free error (B)\n");
	return ;    }
   
	/* CUBLAS Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("\n\t Error :  CUBLAS shutdown error \n");
	return ;    }
	return ;
}


/*
 * void
 * cublasStrsm (char side, char uplo, char transa, char diag, int m, int n,
 *              float alpha, const float *A, int lda, float *B, int ldb)
 *
 * solves one of the matrix equations
 *
 *    op(A) * X = alpha * B,   or   X * op(A) = alpha * B,
 */
void cudablas_strsm(const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const float alpha, const float *host_A, const int lda,
                 float *host_B, const int ldb)
{    
	cublasStatus status;
	float *dev_A, *dev_B;
	int i;
	char side, uplo, transA, diag;
	unsigned long long sizeA, sizeB;
	
	if(Side == CblasLeft)
		side = 'L';
	if(Side == CblasRight)
		side = 'R';
		
	if(Uplo == CblasUpper)
		uplo = 'U';
	if(Uplo == CblasUpper)
		uplo = 'L';
		
	if(TransA == CblasNoTrans)
		transA = 'N';
	if(TransA == CblasTrans)
		transA = 'T';
		
	if(Diag == CblasUnit)
		diag = 'U';
	if(Diag == CblasNonUnit)
		diag = 'N';
		
	/* Calculate the matrx size */
	sizeA = sizeof(host_A);
	sizeB = M*N;	

	/* Initialize CUBLAS */
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  CUBLAS initialization error.\n");
	return ;    }

	
	/* Allocate device memory for the matrices */
	status = cublasAlloc(sizeA, sizeof(dev_A[0]), (void**)&dev_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memeory on device (A)\n");
	return ;    }
	
	status = cublasAlloc(sizeB, sizeof(dev_B[0]), (void**)&dev_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Failed to allocate memory on device (B) \n");
	return ;    }

	/* Initialize the device matrices with the host matrices */
	status = cublasSetVector(sizeA, sizeof(host_A[0]), host_A, 1, dev_A, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write A on dev failed)\n");
	return ;   }

	status = cublasSetVector(sizeB, sizeof(host_B[0]), host_B, 1, dev_B, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (write B on dev failed)\n");
	return ;    }

	/* Clear last error */
	cublasGetError();

	/* Performs operation (cblasStrsm) using cublas */
	
	cublasStrsm(side, uplo, transA, diag, M, N, alpha, dev_A, lda, dev_B, ldb);

	status = cublasGetError();
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : kernel execution error on device .\n");
	return ;    }

	    
	/* Read the result from dev to host  */
	status = cublasGetVector(sizeB, sizeof(float), dev_B, 1, host_B, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Device access error (read C from dev failed)\n");
	return ;    }

	/* Memory clean up */
	
	status = cublasFree(dev_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error : Memory free error (A)\n");
	return ;    }

	status = cublasFree(dev_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
	printf ("\n\t Error :  memory free error (B)\n");
	return ;    }
   
	/* CUBLAS Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("\n\t Error :  CUBLAS shutdown error \n");
	return ;    }
	return ;
}

