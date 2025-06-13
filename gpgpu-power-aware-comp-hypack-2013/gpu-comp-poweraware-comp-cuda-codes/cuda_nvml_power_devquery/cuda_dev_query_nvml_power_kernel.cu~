/*
	Objective : To launch deviceQuery to measure startup power of a GPU
	program   : run deviceQuery on each device	 
	author 	  : HPC-FTEG 
*/

#include<cuda_dev_query_nvml_power_kernel_functions.h>
/**
*  This function open a file devicequery.txt in current directory
*  count no of devices present on system
*  execute deviceQuery on each system
**/

void *deviceQueryFunc(void *t)
{

    	FILE *fp;
	fp = fopen("./devicequery.txt", "a");
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

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


    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        fprintf(fp,"\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        fprintf(fp,"  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        fprintf(fp,"  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        char msg[256];
        sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
        fprintf(fp,"%s", msg);

        fprintf(fp,"  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


#if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        fprintf(fp,"  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
        fprintf(fp,"  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize)
        {
            fprintf(fp,"  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
        }
#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
        int memoryClock;
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        fprintf(fp,"  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        int memBusWidth;
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        fprintf(fp,"  Memory Bus Width:                              %d-bit\n", memBusWidth);
        int L2CacheSize;
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if (L2CacheSize)
        {
            fprintf(fp,"  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
        }
#endif

        fprintf(fp,"  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
               deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
               deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        fprintf(fp,"  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

        fprintf(fp,"  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        fprintf(fp,"  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        fprintf(fp,"  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        fprintf(fp,"  Warp size:                                     %d\n", deviceProp.warpSize);
        fprintf(fp,"  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        fprintf(fp,"  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        fprintf(fp,"  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        fprintf(fp,"  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        fprintf(fp,"  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
        fprintf(fp,"  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
        fprintf(fp,"  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        fprintf(fp,"  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        fprintf(fp,"  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
        fprintf(fp,"  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        fprintf(fp,"  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
        fprintf(fp,"  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#ifdef WIN32
        fprintf(fp,"  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
        fprintf(fp,"  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
        fprintf(fp,"  Device PCI Bus ID / PCI location ID:           %d / %d\n", deviceProp.pciBusID, deviceProp.pciDeviceID);

    }

	cudaDeviceReset();
        /* sleep */
        sleep(60);

        /* send signal to other thread to stop power measurement. */
        sigFlag = 0;
	

    // finish
    exit(EXIT_SUCCESS);
}
