/*****************************************************************************

                         C-DAC Tech Workshop :  HeGaPa-2012
                              July 16 - 20, 2012

  Example     :  simple_kernel_params.cu
 
  Objective   : Write a CUDA  program to understand parameter passing.                 

  Input       : None 

  Output      : addition of the given data. 
                                                                                                                            
  Created     : May-2012    

  E-mail      : betatest@cdac.in         
                                 
****************************************************************************/



#include <stdio.h>
#include<cuda.h>

__global__ void add( int a, int b, int *c ) {
    *c = a + b;
}

/*Utility macro CUDA SAFE CALL */
void CUDA_SAFE_CALL(cudaError_t call)
{
        cudaError_t ret = call;
        //printf("RETURN FROM THE CUDA CALL:%d\t:",ret);                                        
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
    int c;
    int *dev_c;
    CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_c, sizeof(int) ) );

    add<<<1,1>>>( 2, 7, dev_c );

    CUDA_SAFE_CALL( cudaMemcpy( &c, dev_c, sizeof(int),
                              cudaMemcpyDeviceToHost ) );
    printf( "2 + 3 = %d\n", c );
    CUDA_SAFE_CALL( cudaFree( dev_c ) );

    return 0;
}
