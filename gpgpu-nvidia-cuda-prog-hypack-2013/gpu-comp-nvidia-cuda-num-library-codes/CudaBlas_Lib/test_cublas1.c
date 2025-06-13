#include <stdio.h>
#include <stdlib.h>
#include "cudablas.h"
//void cudablas_sscal (int n, float alpha, float *hx, int incx);

int main( int argc, char **argv) 
{
    int i, n, pos;
    float *x, *y;

    if( argc != 2) {
        printf( "Usage: executable #vecsize \n");
        return 1;
    }
    n = atoi( argv[1] );

    x = (float *) malloc(n*sizeof(float));
    if( x == NULL )  {
        printf("Error: Cann't allocate memory on local host\n ");
        return 2;
    }

    y = (float *) malloc(n*sizeof(float));
    if( y == NULL )  {
        printf( "Error: Cann't allocate memory on local host \n" );
        return 2;
    }

    for( i = 0; i < n; i++) x[i] = 1.0*(i+1);

//  pos = cudablas_isamax (n, x, 1);
    for( i = 0; i < n; i++) printf( "%f \n", x[i] );

    cudablas_sscal (n, 2.0, x, 1);

    printf("\n");

    for( i = 0; i < n; i++) printf( "%f \n", x[i] );

    return 0;
}

