
/******************************************************************

	C-DAC Tech Workshop : hyPACK-2013
                 October 15-18, 2013

  Example     :  vector_vector_addition.cu

  Objective   : Write a CUDA Program to perform vector addition.

  Input       : None

  Output      : addition result

  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

*******************************************************************/

#include<stdio.h>
#include<cuda.h>

#define N   10

__global__ void add( int *a, int *b, int *c ) {
    int tid = blockIdx.x;    // this thread handles the data at its thread id
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

/*cuda safe call*/
void CUDA_SAFE_CALL(cudaError_t call)
{
        cudaError_t ret = call;
        switch(ret)
        {
                case cudaSuccess:
                                break;
                default:
                        {
                                printf(" ERROR at line :%i.%d' ' %s\n",__LINE__,ret,cudaGetErrorString(ret));
                                exit(-1);
                                break;
                        }
        }
}


int main( void ) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // allocate the memory on the GPU
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_b, N * sizeof(int) ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_c, N * sizeof(int) ) );

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    CUDA_SAFE_CALL( cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    CUDA_SAFE_CALL( cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );

    add<<<N,1>>>( dev_a, dev_b, dev_c );

    // copy the array 'c' back from the GPU to the CPU
    CUDA_SAFE_CALL( cudaMemcpy( c, dev_c, N * sizeof(int),
                              cudaMemcpyDeviceToHost ) );

    // display the results
    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }

    // free the memory allocated on the GPU
    CUDA_SAFE_CALL( cudaFree( dev_a ) );
    CUDA_SAFE_CALL( cudaFree( dev_b ) );
    CUDA_SAFE_CALL( cudaFree( dev_c ) );

    return 0;
}
