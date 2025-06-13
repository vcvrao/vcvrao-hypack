/*****************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example               : cuda_malloc_test.cu

 Objective             : Objective is to demonstrate the time taken by cudaMalloc 
                         using the pageable host-memory  
 
 Input                 : None

 Output                : Time using cudaMalloc(up,down),                                              
                          copy speed in MB/s(up,down)                                              

 Created               : August-2013

 E-mail                : hpcfte@cdac.in     


*********************************************************************************/

#include "cudaSafeCall.h"

#define SIZE    (64*1024*1024)


float cuda_malloc_test( int size, bool up ) {
    cudaEvent_t     start, stop;
    int             *a, *dev_a;
    float           elapsedTime;

    CUDA_SAFE_CALL( cudaEventCreate( &start ) );
    CUDA_SAFE_CALL( cudaEventCreate( &stop ) );

    a = (int*)malloc( size * sizeof( *a ) );
    CUDA_HANDLE_NULL( a );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_a,
                              size * sizeof( *dev_a ) ) );

    CUDA_SAFE_CALL( cudaEventRecord( start, 0 ) );
    for (int i=0; i<100; i++) {
        if (up)
            CUDA_SAFE_CALL( cudaMemcpy( dev_a, a,
                                  size * sizeof( *dev_a ),
                                  cudaMemcpyHostToDevice ) );
        else
            CUDA_SAFE_CALL( cudaMemcpy( a, dev_a,
                                  size * sizeof( *dev_a ),
                                  cudaMemcpyDeviceToHost ) );
    }
    CUDA_SAFE_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_SAFE_CALL( cudaEventSynchronize( stop ) );
    CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );

    free( a );
    CUDA_SAFE_CALL( cudaFree( dev_a ) );
    CUDA_SAFE_CALL( cudaEventDestroy( start ) );
    CUDA_SAFE_CALL( cudaEventDestroy( stop ) );

    return elapsedTime;
}



int main( void ) {
    float           elapsedTime;
    float           MB = (float)100*SIZE*sizeof(int)/1024/1024;


    // try it with cudaMalloc
    elapsedTime = cuda_malloc_test( SIZE, true );
    printf( "Time using cudaMalloc:  %3.1f ms\n",
            elapsedTime );
    printf( "\tMB/s during copy up:  %3.1f\n",
            MB/(elapsedTime/1000) );

    elapsedTime = cuda_malloc_test( SIZE, false );
    printf( "Time using cudaMalloc:  %3.1f ms\n",
            elapsedTime );
    printf( "\tMB/s during copy down:  %3.1f\n",
            MB/(elapsedTime/1000) );
}

