#include <stdio.h>

#include <cudablas.h>


int main(int argc, char **argv)
{
     int i, j, m,n,k, lda, ldb, ldc;
     float *A, *B, *C;

     enum CBLAS_TRANSPOSE transA = CblasNoTrans;
     enum CBLAS_TRANSPOSE transB = CblasNoTrans;

     float alpha = 1.0, beta = 1.0;

     m = 4; 
     n = 4;
     k = 4;

     lda = m;
     ldb = k;
     ldc = m;

     A = ( float *) malloc( m*k*sizeof(float) );
     B = ( float *) malloc( k*n*sizeof(float) );
     C = ( float *) malloc( m*n*sizeof(float) );

     for( i = 0; i < m*k; i++) A[i] = 1;
     for( i = 0; i < k*n; i++) B[i] = 1;
     for( i = 0; i < m*n; i++) C[i] = 1;
    
     cudablas_sgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

     for( i = 0; i < m; i++)
     {
	printf("\n");
	for(j=0; j < n; j++)
          printf( " %f \t", C[i*m + j]);
     }

     printf("\n");

     return 1;
}
