


#include<cuda_globalmemory_nvml_power_kernel_define.h>
#include<cublas.h>

#define VECT_SIZE 10000000
#define BLOCK_SIZE 512

/**
 * conditional variable 
**/
int sigFlag = 1;

/**
 * holds source array and destination array
**/
float *srcArray , *destArray;
float3 *srcArray3; // the array of float3 datatype

/** 
 * Holds total time taken in launching the kernel 
 * and bandwidth of accessing global memory in 
 * colalesced manner
**/
float elapsedTimes, bandWidths;

/** 
 * holds start & end time
**/
cudaEvent_t start,stop;

/**
 * Holds cure APIs return value.
**/
cudaError_t err = cudaSuccess;
        
double bytes = 2 * sizeof(float) * VECT_SIZE;

/**
*handle error
**/
void HANDLE_ERROR(cudaError_t call)
{
        cudaError_t ret = call;
        //printf("RETURN FROM THE CUDA CALL:%d\t:",ret);                                        
        switch(ret)
        {
                case cudaSuccess:
                //              printf("Success\n");                    
                                break;
              case cudaErrorInvalidValue:                             
                                {
                                printf("ERROR: InvalidValue:%i.\n",__LINE__);
                                exit(-1);
                                break;  
                                }                       
                case cudaErrorInvalidDevicePointer:                     
                                {
                                printf("ERROR:Invalid Device pointeri:%i.\n",__LINE__);
                                exit(-1);
                                break;
                                }                       
                case cudaErrorInvalidMemcpyDirection:                   
                                {
                                printf("ERROR:Invalid memcpy direction:%i.\n",__LINE__);        
                                exit(-1);
                                break;
                                }                  
                default:
                        {
                                printf(" ERROR at line :%i.%d' ' %s\n",__LINE__,ret,cudaGetErrorString(ret));
                                exit(-1);
                                break;
                        }
        }
}
