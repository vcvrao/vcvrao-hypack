
/**************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                         October 15-18, 2013

 Example             : cudaHostAllocTest.cu

 Objective           : Objective is to demonstrate the time taken by cudaMalloc 
                         using the pinned host-memory  
 
 Input               : None

 Output              : Time using cudaHostAlloc(up,down),                                              
                         copy speed in MB/s(up,down)                                              

 Created             : August-2013

 E-mail              : hpcfte@cdac.in     

*************************************************************************/

#include "cudaSafeCall.h"

#define SIZE    (64*1024*1024)
float cuda_host_alloc_test( int size, bool up ) {
    cudaEvent_t     start, stop;
    int             *a, *dev_a;
    float           elapsedTime;

    CUDA_SAFE_CALL( cudaEventCreate( &start ) );
    CUDA_SAFE_CALL( cudaEventCreate( &stop ) );

    CUDA_SAFE_CALL( cudaHostAlloc( (void**)&a,
                                 size * sizeof( *a ),
                                 cudaHostAllocDefault ) );
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_a,
                              size * sizeof( *dev_a ) ) );

     CUDA_SAFE_CALL( cudaEventRecord( start, 0 ) );
    for (int i=0; i<100; i++) {
        if (up)
            CUDA_SAFE_CALL( cudaMemcpy( dev_a, a,
                                  size * sizeof( *a ),
                                  cudaMemcpyHostToDevice ) );
        else
            CUDA_SAFE_CALL( cudaMemcpy( a, dev_a,
                                  size * sizeof( *a ),
                                  cudaMemcpyDeviceToHost ) );
    }
    CUDA_SAFE_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_SAFE_CALL( cudaEventSynchronize( stop ) );
    CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );

    CUDA_SAFE_CALL( cudaFreeHost( a ) );
    CUDA_SAFE_CALL( cudaFree( dev_a ) );
    CUDA_SAFE_CALL( cudaEventDestroy( start ) );
    CUDA_SAFE_CALL( cudaEventDestroy( stop ) );

    return elapsedTime;
}

int main( void ) {
    float           elapsedTime;
    float           MB = (float)100*SIZE*sizeof(int)/1024/1024;
    // now try it with cudaHostAlloc
    elapsedTime = cuda_host_alloc_test( SIZE, true );
    printf( "Time using cudaHostAlloc:  %3.1f ms\n",
            elapsedTime );
    printf( "\tMB/s during copy up:  %3.1f\n",
            MB/(elapsedTime/1000) );

    elapsedTime = cuda_host_alloc_test( SIZE, false );
    printf( "Time using cudaHostAlloc:  %3.1f ms\n",
            elapsedTime );
    printf( "\tMB/s during copy down:  %3.1f\n",
            MB/(elapsedTime/1000) );
}

