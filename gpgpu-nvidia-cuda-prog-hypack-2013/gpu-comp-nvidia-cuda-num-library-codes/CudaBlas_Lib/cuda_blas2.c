#include "cudablas.h"

///////////////////////////////////////////////////////////////////////////////
// Programmer :  Chaman Singh Verma $ Dr.VCV.Rao (C-DAC,Pune )
// Date       :  18th July 208
// Updated    :  August 2013
// 
//	C-DAC Tech Workshop : hyPACK-2013
//            October 15-18, 2013
//
// Place      :  CDAC
// Copyrights :  Nothing, free to modify and download. 
//
///////////////////////////////////////////////////////////////////////////////

int copy_to_device(float **dev_array, int n, const float *hx)
{
  cublasStatus status;
  float *dx;

  status = cublasAlloc(n, sizeof(float), (void**)&dx);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf ("\n\t Error: Memory allocation error on device.\n");
    return 1; 
  }

  status = cublasSetVector(n, sizeof(float), hx, 1, dx, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf ("Error: Transfering data from host to device).\n");
    return 1; 
  }

  *dev_array = dx;
}
///////////////////////////////////////////////////////////////////////////////
int copy_from_device( float *hx, size_t n, float *dx)
{
  cublasStatus status = cublasGetError();

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf ("\n\t Error: CUBLAS execution error.\n");
    return 1; 
  }

  status = cublasGetVector(n, sizeof(float), dx, 1, hx, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf ("Error: Transfering data from device to host).\n");
    return 1; 
  }
  return 0;
}
///////////////////////////////////////////////////////////////////////////////

void cudablas2_initialize( int m, int n, float *x, float *y, float *A)
{

}

///////////////////////////////////////////////////////////////////////////////

int cudablas2_finalize( float *x, float *y, float *A)
{
  cublasStatus  status = cublasFree(x);

  if (status != CUBLAS_STATUS_SUCCESS) 
  {
    printf ("Error : Failed to free the memory on device.\n");
    return 1; 
  }

  status = cublasFree(y);
 if (status != CUBLAS_STATUS_SUCCESS) 
  {
    printf ("\n\t Error : Failed to free the memory on device.\n");
    return 1;    
  }

  status = cublasFree(A);
  if (status != CUBLAS_STATUS_SUCCESS) 
  {
    printf ("\n\t Error : Failed to free the memory on device.\n");
    return 1;  
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////
void cudablas_sgbmv(const enum CBLAS_TRANSPOSE trans, const int m, const int n,
                 const int kl, const int ku, const float alpha,
                 const float *hA, const int lda, const float *hx,
                 const int incx, const float beta, float *hy, const int incy)
{    

  //**************************************************************************
  //   y  = alpha*op(A)*x + beta*y
  //   where op(A) = A or op(A) = trans(A)
  //**************************************************************************
  float *dx, *dy, *dA;
  char transA;
  
  if(trans == CblasTrans)
  {
	transA = 'T';
  	copy_to_device(&dx, m, hx);
  	copy_to_device(&dy, n, hy);
  }
  if(trans == CblasNoTrans)
  {
	transA = 'N';
  	copy_to_device(&dx, n, hx);
  	copy_to_device(&dy, m, hy);
  }

  copy_to_device(&dA, m*n, hA);

  cublasSgbmv(transA, m, n,  kl, ku, alpha, dA, lda, dx, incx, beta, dy, incy);

  copy_from_device( hy, m, dy );

  cudablas2_finalize(dx, dy, dA);  

}

///////////////////////////////////////////////////////////////////////////////

void cudablas_sgemv(const enum CBLAS_TRANSPOSE trans, const int m, const int n,
                 const float alpha, const float *hA, const int lda,
                 const float *hx, const int incx, const float beta,
                 float *hy, const int incy)
{    

  /////////////////////////////////////////////////////////////////////////////////
  // y = alpha*op(A)*x + beta*y
  //     Where op(A) = A  or op(A) = trans(A)
  ////////////////////////////////////////////////////////////////////////////////

  float *dx, *dy, *dA;
  char transA;
  
  if(trans == CblasTrans)
  {
	transA = 'T';
  	copy_to_device(&dx, m, hx);
  	copy_to_device(&dy, n, hy);
  }
  if(trans == CblasNoTrans)
  {
	transA = 'N';
  	copy_to_device(&dx, n, hx);
  	copy_to_device(&dy, m, hy);
  }

  copy_to_device(&dA, m*n, hA);

  cublasSgemv(transA, m, n, alpha, dA, lda, dx, incx, beta, dy, incy);

  copy_from_device( hy, m, dy );
  cudablas2_finalize(dx, dy, dA);  
}

///////////////////////////////////////////////////////////////////////////////

void cudablas_sger(const int m, const int n, 
		 const float alpha, const float *hx, const int incx,
                 const float *hy, const int incy, float *ha, const int lda)
{
	    
  float *dx, *dy, *dA;

  // **************************************************************************
  //    A = alpha*x*Trans(y) + A 
  // **************************************************************************

  copy_to_device(&dx, m, hx);
  copy_to_device(&dy, n, hy);
  copy_to_device(&dA, m*n, ha);

  cublasSger(m, n, alpha, dx, incx, dy, incy, dA,lda);

  copy_from_device(ha, m*n, dA);
  cudablas2_finalize(dx, dy, dA);  

}
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

void cudablas_ssbmv(const enum CBLAS_UPLO Uplo, const int n, const int k, 
				 const float alpha,const float *ha,
                 const int lda, const float *hx, const int incx,
                 const float beta, float *hy, const int incy)
{
	    
  float *dx, *dy, *dA;

  // **************************************************************************
  //    y = alpha*A*x + beta*y  
  // **************************************************************************

  char uplo;
  
  if(Uplo == CblasUpper)
	uplo = 'U';
  if(Uplo == CblasLower)
	uplo = 'L';

  copy_to_device(&dx, n, hx);
  copy_to_device(&dy, n, hy);
  copy_to_device(&dA, n*n, ha);

  cublasSsbmv(uplo, n, k, alpha, dA, lda, dx, incx, beta, dy, incy);

  copy_from_device( hy, n, dy );
  cudablas2_finalize(dx, dy, dA);  

}
///////////////////////////////////////////////////////////////////////////////

void cudablas_sspmv(const enum CBLAS_UPLO Uplo,
                 const int n, const float alpha, const float *hA,
                 const float *hx, const int incx,
                 const float beta, float *hy, const int incy)
{    
  float *dx, *dy, *dA;

  // **************************************************************************
  //    y = alpha*A*x + beta*y  
  // **************************************************************************


  char uplo;
  
  if(Uplo == CblasUpper)
	uplo = 'U';
  if(Uplo == CblasLower)
	uplo = 'L';
  
  copy_to_device(&dx, n, hx);
  copy_to_device(&dy, n, hy);
  copy_to_device(&dA, n*n, hA);

  cublasSspmv(uplo, n, alpha, dA, dx, incx, beta, dy, incy);

  copy_from_device( hy, n, dy );
  cudablas2_finalize(dx, dy, dA);  


}

///////////////////////////////////////////////////////////////////////////////

void cudablas_sspr(const enum CBLAS_UPLO Uplo,
                const int n, const float alpha, const float *hx,
                const int incx, float *hA)
{    
  float *dx, *dA;

  // **************************************************************************
  //    y = alpha*x*trans(X) +  A; 
  // **************************************************************************

  char uplo;
  
  if(Uplo == CblasUpper)
	uplo = 'U';
  if(Uplo == CblasLower)
	uplo = 'L';
	
  copy_to_device(&dx, n, hx);
  copy_to_device(&dA, n*n, hA);
  
  cublasSspr(uplo, n, alpha, dx, incx, dA);

  copy_from_device( hA, n, dA );
  cudablas2_finalize(dx, NULL, dA);  

}


///////////////////////////////////////////////////////////////////////////////

void cudablas_sspr2(const enum CBLAS_UPLO Uplo,
                const int n, const float alpha, const float *hx,
                const int incx, const float *hy, const int incy, float *hA)
{    
  float *dx, *dy, *dA;

  // **************************************************************************
  //    A = alpha*x*trans(y) + alpha*y*trans(x) + A; 
  // **************************************************************************

  char uplo;
  
  if(Uplo == CblasUpper)
	uplo = 'U';
  if(Uplo == CblasLower)
	uplo = 'L';
	
  copy_to_device(&dx, n, hx);
  copy_to_device(&dy, n, hy);
  copy_to_device(&dA, n*n, hA);

  cublasSspr2(uplo, n, alpha, dx, incx, dy, incy, dA);

  copy_from_device( hA, n*n, dA );
  cudablas2_finalize(dx, dy, dA);  


}

///////////////////////////////////////////////////////////////////////////////

void cudablas_ssymv(const enum CBLAS_UPLO Uplo,
                 const int n, const float alpha, const float *hA,
                 const int lda, const float *hx, const int incx,
                 const float beta, float *hy, const int incy)
{    

  // *************************************************************************
  //    y = alpha*A*x + beta*y
  // *************************************************************************

  float *dx, *dy, *dA;

  char uplo;
  
  if(Uplo == CblasUpper)
	uplo = 'U';
  if(Uplo == CblasLower)
	uplo = 'L';
	
  copy_to_device(&dx, n, hx);
  copy_to_device(&dy, n, hy);
  copy_to_device(&dA, n*n, hA);
  
  cublasSsymv(uplo, n, alpha, dA, lda, dx, incx, beta, dy, incy);

  copy_from_device( hy, n, dy );
  cudablas2_finalize(dx, dy, dA);  

}

///////////////////////////////////////////////////////////////////////////////

void cudablas_ssyr(const enum CBLAS_UPLO Uplo,
                const int n, const float alpha, const float *hx,
                const int incx, float *hA, const int lda)
{    
  float *dx, *dA;

  // *************************************************************************
  //  A = alpha*x*trans(x) + A
  // *************************************************************************

  char uplo;
  
  if(Uplo == CblasUpper)
	uplo = 'U';
  if(Uplo == CblasLower)
	uplo = 'L';
	
  copy_to_device(&dx, n, hx);
  copy_to_device(&dA, n*n, hA);

  cublasSsyr(uplo, n, alpha, dx, incx, dA, lda);

  copy_from_device( hA, n, dA );
  cudablas2_finalize(dx, NULL,  dA);  

}

///////////////////////////////////////////////////////////////////////////////

void cudablas_ssyr2(const enum CBLAS_UPLO Uplo,
                const int n, const float alpha, const float *hx,
                const int incx, const float *hy, const int incy, float *hA,
                const int lda)

{    
  float *dx, *dy, *dA;

  // *************************************************************************
  // A = alpha*x*trans(y) + alpha*y*trans(x) + A
  // *************************************************************************

  char uplo;
  
  if(Uplo == CblasUpper)
	uplo = 'U';
  if(Uplo == CblasLower)
	uplo = 'L';
	
  copy_to_device(&dx, n, hx);
  copy_to_device(&dy, n, hy);
  copy_to_device(&dA, n*n, hA);
  
  cublasSsyr2(uplo, n, alpha, dx, incx, dy, incy, dA, lda);

  copy_from_device( hA, n, dA );
  cudablas2_finalize(dx, dy, dA);  

}

///////////////////////////////////////////////////////////////////////////////


void cudablas_stbmv(const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE transA, const enum CBLAS_DIAG Diag,
                 const int n, const int k, const float *hA, const int lda, 
                 float *hx, const int incx)
{    
  float *dx, *dA;

  // *************************************************************************
  //  x = op(A)*x :   Where op(A) = trans(A)
  // *************************************************************************

  char uplo, trans, diag;
  
  if(Uplo == CblasUpper)
	uplo = 'U';
  if(Uplo == CblasLower)
	uplo = 'L';
		 
  if(transA == CblasTrans)
	trans = 'T';
  if(transA == CblasNoTrans)
	trans = 'N';
		 
  if(Diag == CblasUnit)
	diag = 'U';
  if(Diag == CblasNonUnit)
	diag = 'N';
	
  copy_to_device(&dx, n, hx);
  copy_to_device(&dA, n*n, hA);

  cublasStbmv(uplo, trans, diag, n, k, dA, lda, dx, incx);

  copy_from_device( hx, n, dx );
  cudablas2_finalize(dx, NULL, dA);  


}

///////////////////////////////////////////////////////////////////////////////

void cudablas_stbsv(const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE transA, const enum CBLAS_DIAG Diag,
                 const int n, const int k, const float *hA, const int lda,
                 float *hx, const int incx)
{    
  float *dx, *dA;

  // *************************************************************************
  // op(A)*x = b;  where op(A) = A or op(A) = trans(A)
  // *************************************************************************

  char uplo, trans, diag;
  
  if(Uplo == CblasUpper)
	uplo = 'U';
  if(Uplo == CblasLower)
	uplo = 'L';
		 
  if(transA == CblasTrans)
	trans = 'T';
  if(transA == CblasNoTrans)
	trans = 'N';
		 
  if(Diag == CblasUnit)
	diag = 'U';
  if(Diag == CblasNonUnit)
	diag = 'N';
	
  copy_to_device(&dx, n, hx);
  copy_to_device(&dA, n*n, hA);
  
  cublasStbsv(uplo, trans, diag, n, k, dA, lda, dx, incx);

  copy_from_device( hx, n, dx );
  cudablas2_finalize(dx, NULL, dA);  

}

///////////////////////////////////////////////////////////////////////////////

void cudablas_stpmv(const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE transA, const enum CBLAS_DIAG Diag,
                 const int n, const float *hA, float *hx, const int incx)
{    

  float  *dx, *dA;

  // *************************************************************************
  // x = op(A)*x ;  where op(A) = A or op(A) = trans(A)
  // *************************************************************************

  char uplo, trans, diag;
  
  if(Uplo == CblasUpper)
	uplo = 'U';
  if(Uplo == CblasLower)
	uplo = 'L';
		 
  if(transA == CblasTrans)
	trans = 'T';
  if(transA == CblasNoTrans)
	trans = 'N';
		 
  if(Diag == CblasUnit)
	diag = 'U';
  if(Diag == CblasNonUnit)
	diag = 'N';
	
  copy_to_device(&dx, n, hx);
  copy_to_device(&dA, n*n, hA);
  
  cublasStpmv(uplo, trans, diag, n, dA, dx, incx);

  copy_from_device( hx, n, dx );
  cudablas2_finalize(dx, NULL, dA);  

}

///////////////////////////////////////////////////////////////////////////////

void cudablas_stpsv(const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int n, const float *hA, float *hx, const int incx)
{    
  float *dx, *dA;

  // *************************************************************************
  // op(A)*x = b;   op(A) = A or op(A) = trans(A)
  // *************************************************************************

  char uplo, trans, diag;
  
  if(Uplo == CblasUpper)
	uplo = 'U';
  if(Uplo == CblasLower)
	uplo = 'L';
		 
  if(TransA == CblasTrans)
	trans = 'T';
  if(TransA == CblasNoTrans)
	trans = 'N';
		 
  if(Diag == CblasUnit)
	diag = 'U';
  if(Diag == CblasNonUnit)
	diag = 'N';
	
  copy_to_device(&dx, n, hx);
  copy_to_device(&dA, n*n, hA);
  
  cublasStpsv(uplo, trans, diag, n, dA, dx, incx);

  copy_from_device( hx, n, dx );
  cudablas2_finalize(dx, NULL, dA);  

}

///////////////////////////////////////////////////////////////////////////////


void cudablas_strmv(const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int n, const float *hA, const int lda, 
                 float *hx, const int incx)
{    
  float *dx, *dA;

  // *************************************************************************
  // x = op(A)*x;   op(A) = A or op(A) = trans(A)
  // *************************************************************************

  char uplo, trans, diag;
  
  if(Uplo == CblasUpper)
	uplo = 'U';
  if(Uplo == CblasLower)
	uplo = 'L';
		 
  if(TransA == CblasTrans)
	trans = 'T';
  if(TransA == CblasNoTrans)
	trans = 'N';
		 
  if(Diag == CblasUnit)
	diag = 'U';
  if(Diag == CblasNonUnit)
	diag = 'N';
	
  copy_to_device(&dx, n, hx);
  copy_to_device(&dA, n*n, hA);

  cublasStrmv(uplo, trans, diag, n, dA, lda, dx, incx);

  copy_from_device( hx, n, dx );
  cudablas2_finalize(dx, NULL, dA);  

}

///////////////////////////////////////////////////////////////////////////////

void cudablas_strsv(const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int n, const float *hA, const int lda, float *hx,
                 const int incx)
{    
  // *************************************************************************
  // op(A)*x = b ;   op(A) = A or op(A) = trans(A)
  // *************************************************************************

  float *dx, *dA;
  char uplo, trans, diag;
  
  if(Uplo == CblasUpper)
	uplo = 'U';
  if(Uplo == CblasLower)
	uplo = 'L';
		 
  if(TransA == CblasTrans)
	trans = 'T';
  if(TransA == CblasNoTrans)
	trans = 'N';
		 
  if(Diag == CblasUnit)
	diag = 'U';
  if(Diag == CblasNonUnit)
	diag = 'N';
	
  copy_to_device(&dx, n, hx);
  copy_to_device(&dA, n, hA);

  cublasStrsv(uplo, trans, diag, n, dA, lda, dx, incx);

  copy_from_device( hx, n, dx );
  cudablas2_finalize(dx, NULL, dA);  

}
///////////////////////////////////////////////////////////////////////////////
