#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "cublas.h"

int copy_to_device(float **dev_array, int n, const float *hx)
{
  float *dx;
  cublasStatus status;

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

  return 0;
}
///////////////////////////////////////////////////////////////////////////////
int copy_from_device( float *hx, int n, const float *dx)
{
  cublasStatus status;

  status = cublasGetError();

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf ("\n\t Error: CUBLAS execution error.\n");
    return 1; 
  }
  assert( dx );

  status = cublasGetVector(n, sizeof(float), dx, 1, hx, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf ("Error: Transfering data from device to host).\n");
    return 1; 
  }
  return 0;
}
///////////////////////////////////////////////////////////////////////////////

int cudablas1_finalize( float *x, float *y)
{
  cublasStatus status;

  if( x ) {
      status = cublasFree(x);
      if (status != CUBLAS_STATUS_SUCCESS) {
           printf ("Error : Failed to free the memory on device.\n");
           return 1;    
      }
  }

  if( y ) {
      status = cublasFree(y);
      if (status != CUBLAS_STATUS_SUCCESS) {
          printf ("Error : Failed to free the memory on device.\n");
          return 1;    
      }
   }
   return 0;
}

///////////////////////////////////////////////////////////////////////////////
#ifdef fdsfsdfs

void
cudablas_saxpy (int n, float alpha, const float *hx, int incx, float *hy,
		int incy)
{    
  float *dx, *dy;

  copy_to_device( &dx, n, hx);
  copy_to_device( &dy, n, hy);

  cublasSaxpy(n, alpha, dx, incx, dy, incy);

  copy_from_device( hy, n, dy);
        
  cudablas1_finalize( dx, dy );
}

///////////////////////////////////////////////////////////////////////////////

CBLAS_INDEX cudablas_isamax (int n, const float *hx, int incx)
{    
  float *dx = NULL;
  int i;

  CBLAS_INDEX pos;
  copy_to_device(&dx, n, hx);

  pos = cublasIsamax(n, dx, incx);

  cudablas1_finalize( dx, NULL);

  return pos;
}
///////////////////////////////////////////////////////////////////////////////

CBLAS_INDEX cudablas_isamin (int n, const float *hx, int incx)
{    
  float *dx;
  CBLAS_INDEX pos;

  copy_to_device( &dx, n, hx);

  pos = cublasIsamin(n, dx, incx);

  cudablas1_finalize( dx, NULL);

  return pos;
}
///////////////////////////////////////////////////////////////////////////////

float cudablas_sasum (int n, const float *hx, int incx)
{
  float  *dx, val;
  copy_to_device( &dx, n, hx);

  val = cublasSasum(n, dx, incx);

  cudablas1_finalize( dx, NULL);

  return val;
}
///////////////////////////////////////////////////////////////////////////////

float
cudablas_sdot (int n, const float *hx, int incx, const float *hy, int incy)
{    
  float  *dx, *dy, dprod;

  copy_to_device( &dx, n, hx);
  copy_to_device( &dy, n, hy);

  dprod = cublasSdot(n, dx, incx, dy, incy);

  cudablas1_finalize( dx, dy);

  return dprod;
}
///////////////////////////////////////////////////////////////////////////////

void
cudablas_scopy (int n, const float *hx, int incx, float *hy, int incy)
{    
  float *dx, *dy;

  copy_to_device( &dx, n, hx);
  copy_to_device( &dy, n, hy);

  cublasScopy(n, dx, incx, dy, incy);

  copy_from_device(hy, n, dy);

  cudablas1_finalize(dx,dy);
}
///////////////////////////////////////////////////////////////////////////////
 
float
cudablas_snrm2 (int n, const float *hx, int incx)
{    
  float *dx;
  float enorm = 0.0f;
  copy_to_device( &dx, n, hx);

  enorm = cublasSnrm2(n, dx, incx);

  cudablas1_finalize(dx,NULL);

  return enorm; 
}
///////////////////////////////////////////////////////////////////////////////

void
cudablas_sswap (int n, float *hx, int incx, float *hy, int incy)
{
  float *dx, *dy;
  copy_to_device( &dx, n, hx);
  copy_to_device( &dy, n, hy);

  cublasSswap(n, dx, incx, dy, incy);
  copy_from_device(hy, n, dy);
  cudablas1_finalize(dx,dy);
}

///////////////////////////////////////////////////////////////////////////////

void
cudablas_srot (int n, float *hx, int incx, float *hy, int incy, float sc, float ss)
{    
  float *dx, *dy;

  copy_to_device( &dx, n, hx);
  copy_to_device( &dy, n, hy);

  cublasSrot(n, dx, incx, dy, incy, sc, ss);

  copy_from_device( hx, n, dx);
  copy_from_device( hy, n, dy);

  cudablas1_finalize(dx,dy);
}
///////////////////////////////////////////////////////////////////////////////
#endif

void
cudablas_sscal (int n, float alpha, float *hx, int incx)
{    
  float *dx = NULL;
  int i, stat = 0;

  stat = copy_to_device( &dx, n, hx);  assert(stat == 0); assert( dx != NULL );

  //printf( " Hello Alpha %f \n", alpha );

  cublasSscal(n, alpha, dx, incx); 

  stat = copy_from_device(hx,n,dx);     assert(stat == 0);
  //for( i = 0; i < n; i++) 
    //printf( "%f \n", hx[i] );

  stat = cudablas1_finalize(dx,NULL);      assert(stat == 0);
}
///////////////////////////////////////////////////////////////////////////////

