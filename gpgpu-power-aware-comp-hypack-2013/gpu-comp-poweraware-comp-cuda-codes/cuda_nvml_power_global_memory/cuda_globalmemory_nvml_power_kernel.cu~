/*
        Objective : To launch blank kernel to measure startup power of a GPU     
        author    : HPC-FTEG 
*/

#include<cuda_globalmemory_nvml_power_kernel_functions.h>
#include<cublas.h>


/**
 * Kernel for setting the array with given element 
 * @param array in which values needs to fill
 * @param constant value 
 * @param vector size 
**/
__global__ void setArray(float *array,  float value, int size)
{
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size)
                array[idx] = value;
}


/**
 * Kernel for accessing global memory in coalesced way.
 * @param destination array
 * @param src array
 * @param vector size 
**/
__global__ void coalescedGMAccess(float* dest,float* src,long size)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < size)
                dest[idx] = src[idx];
}


/**
 * Main function to launch kernel to access global memory
 * @param threadID
**/
void *CoalescedGlobalMemAccessFunc(void *t)
{
        int i;
        CUcontext  context;
        CUdevice   device;

     //calculating number of devices present in system
        int deviceCount = 0,dev;
        cudaError_t error_id=cudaGetDeviceCount(&deviceCount);

        if (error_id != cudaSuccess)
        {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
        }

    // This function call returns 0 if there are no CUDA capable devices.
       if (deviceCount == 0)
       {
        printf("There are no available device(s) that support CUDA\n");
       }

        //finding the 1D grid size 
        int gridSize = VECT_SIZE/BLOCK_SIZE;

        if( VECT_SIZE % BLOCK_SIZE != 0 )
                        gridSize += 1;
    //executing kernel on each devices for 2500 times 
      for (dev = 0; dev < 1; ++dev)
      {
        cudaSetDevice(dev);
        //selecting perticular device
       //For each device create a context explicitly
        cudaDeviceReset();
        CUresult err = cuInit(dev);
        cuDeviceGet(&device, dev);
        err = cuCtxCreate(&context, 0, device);
        if (err != CUDA_SUCCESS)
        {
                fprintf(stderr, "* Error initializing the CUDA context.\n");
                cuCtxDetach(context);
                exit(-1);
        }


        // Set focus on the specified CUDA context
        CUresult cu_status = cuCtxPushCurrent(context);
        if(cu_status != CUDA_SUCCESS)
        {
        fprintf(stderr,"Cannot push current context for device %d, status = %d\n",dev, cu_status);
        return 0;
        }

       // event creation, which will be used for timing the code
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start,0));


        // allocating the memory for the two arrays on the selected device
        // the memory allocated using cudaMalloc will give the memory address that is aligned (128)
        HANDLE_ERROR(cudaMalloc((void **)&srcArray,VECT_SIZE*sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **)&destArray,VECT_SIZE*sizeof(float)));

        // intializing the arrays on the selected device
        for(i = 0;i < 2500 ; i++)
        {
                setArray <<< gridSize,BLOCK_SIZE >>> (srcArray,1.0f,VECT_SIZE);
                setArray <<< gridSize,BLOCK_SIZE >>> (destArray,0.0f,VECT_SIZE);
        }

        //Lunching kernel on selected device
        for(i = 0; i < 2500; i++)
        {
                coalescedGMAccess <<< gridSize,BLOCK_SIZE >>> (destArray,srcArray,VECT_SIZE);
        }
        // Removing arrays from selected device
        HANDLE_ERROR(cudaFree(srcArray));
        HANDLE_ERROR(cudaFree(destArray));
        //stopping timing event
        HANDLE_ERROR(cudaEventRecord(stop,0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        HANDLE_ERROR(cudaEventElapsedTime(&elapsedTimes,start,stop));
        bandWidths = bytes/elapsedTimes;

         /** Free cuda events */
        HANDLE_ERROR(cudaEventDestroy(start));
        HANDLE_ERROR(cudaEventDestroy(stop));


        // Pop the previously pushed CUDA context out of this thread.
        CUresult cu_status1 =cuCtxPopCurrent(&context);
        if(cu_status1 != CUDA_SUCCESS)
        {
        fprintf(stderr,"Cannot push current context for device %d, status = %d\n",dev, cu_status1);
        return 0;
        }


      }
        /* delete all cuda context */
        cuCtxDetach(context);
        /*set set at stable state */
        HANDLE_ERROR(cudaDeviceReset());

        sleep(25);

        /* send signal to other thread to stop power measurement. */
        sigFlag = 0;
        printf("\n\n \t Bandwidth : %f, \t MemAccess Time : %f\n",bandWidths/1000000,elapsedTimes);

        pthread_exit(NULL);
//      return 0;

}

