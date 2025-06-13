
/*************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                       October 15-18, 2013

  Example     :  cuda-find-device.cu
 
  Objective   : Write a CUDA  program to set the gpu .                 

  Input       : None 

  Output      : information about device. 

  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

***********************************************************************/

#include <stdio.h> 
#include <time.h> 
#include <cuda.h> 

/* Utility Macro : CUDA SAFE CALL */
void CUDA_SAFE_CALL( cudaError_t call)
{

    cudaError_t ret = call;
    switch(ret)
    {

        case cudaSuccess:
              break;
        default :
              {

            printf(" ERROR at line :%i.%d' ' %s\n",
            __LINE__,ret,cudaGetErrorString(ret));
            exit(-1);
            break;

             }

    }

}

int main ( void ) {

    int count;
    int dev;
    cudaDeviceProp prop;

    CUDA_SAFE_CALL(cudaGetDeviceCount( &count) );

    for(int i = 0; i < count; i++) {

        CUDA_SAFE_CALL( cudaGetDeviceProperties( &prop, i) );

        CUDA_SAFE_CALL( cudaGetDevice(&dev) );

        printf("Information about the device \t: %d\n", count);

        printf("Name \t\t\t\t: %s\n",prop.name);

        printf("ID of the device : %d\n", dev);
	memset(&prop, 0, sizeof (cudaDeviceProp));

        prop.major = 1;
        prop.minor = 3;
        CUDA_SAFE_CALL( cudaChooseDevice(&dev, &prop ) );

        printf("ID of CUDA device closest to revision 1.3 :%d \n", dev);

        CUDA_SAFE_CALL( cudaSetDevice(dev) );

    }
    return 0;

}
