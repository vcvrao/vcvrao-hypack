#include <stdio.h>

#include <cudablas.h>


int main(int argc, char **argv)
{
     int i, m,n, lda, incx = 1, incy = 1;
     float *A, *x, *y;

     enum CBLAS_TRANSPOSE trans = CblasNoTrans;

     float alpha = 1.0, beta = 0.0;

     m = 5000; 
     n = 5000;
     lda = m;
     A = ( float *) malloc( m*n*sizeof(float) );
     x = ( float *) malloc( m*sizeof(float) );
     y = ( float *) malloc( m*sizeof(float) );

     for( i = 0; i < m*n; i++) A[i] = 1;
     for( i = 0; i < m; i++) x[i] = i;
    
     cudablas_sgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);

     for( i = 0; i < m; i++)
          printf( " %f \n", y[i] );
}
