
#include <stdio.h>
#include <nvml.h>
#include<stdlib.h>
const char * convertToComputeModeString(nvmlComputeMode_t mode)
{
    switch (mode)
    {
        case NVML_COMPUTEMODE_DEFAULT:
            return "Default";
        case NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
            return "Exclusive_Thread";
        case NVML_COMPUTEMODE_PROHIBITED:
            return "Prohibited";
        case NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
            return "Exclusive Process";
        default:
            return "Unknown";
    }
}


int main()
{

    nvmlReturn_t result;
    unsigned int device_count, i,unitCount;

    //STEP-1: First initialize NVML library
    result = nvmlInit();
    if (NVML_SUCCESS != result)
    { 
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));

        printf("Press ENTER to continue...\n");
        getchar();
        return 1;
    }


    //STEP-2:counting how many devices present in syatem
        result = nvmlDeviceGetCount(&device_count);
        if (NVML_SUCCESS != result)
        { 
        printf("Failed to query device count: %s\n", nvmlErrorString(result));
        goto Error;
        }
        printf("Found %d device%s\n\n",device_count, device_count != 1 ? "s" : "");

        //addressing each devices
        printf("Listing devices:\n"); 
        printf("----------------------------------------------------------------------------\n");
  
     for (i = 0; i < device_count; i++)
      {
	nvmlDevice_t device;    //for device identification
	char name[NVML_DEVICE_NAME_BUFFER_SIZE]; //for device name
	nvmlPciInfo_t pci;//for pci information
	nvmlMemory_t meminfo; //for gpu memory information    

	
	//STEP-3:
	// Query for device handle to perform operations on a device
        // You can also query device handle by other features like:
        // nvmlDeviceGetHandleBySerial
        // nvmlDeviceGetHandleByPciBusId
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (NVML_SUCCESS != result)
        { 
            printf("Failed to get handle for device %i: %s\n", i, nvmlErrorString(result));
            goto Error;
        }
	//geting each device name
	result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (NVML_SUCCESS != result)
        { 
            printf("Failed to get name of device %i: %s\n", i, nvmlErrorString(result));
            goto Error;
        }
	// pci.busId is very useful to know which device physically you're talking to
        // Using PCI identifier you can also match nvmlDevice handle to CUDA device.
        result = nvmlDeviceGetPciInfo(device, &pci);
        if (NVML_SUCCESS != result)
        { 
            printf("Failed to get pci info for device %i: %s\n", i, nvmlErrorString(result));
            goto Error;
        }

        printf("%d. %s [%s]\n", i, name, pci.busId);
	
        //calculating gpu global memory information
        result = nvmlDeviceGetMemoryInfo( device, &meminfo );

        if ( NVML_SUCCESS != result ) 
         {    
                printf( "something went wrong %s\n", nvmlErrorString(result));
		return 0;
         }    
        printf("Total installed FB memory (in bytes)=%llu\n",meminfo.total);
	printf("Unallocated FB memory (in bytes).=%llu\n",meminfo.free);
	printf("Allocated FB memory (in bytes). Note that the driver/GPU always sets aside a small amount of memory for bookkeeping=%llu\n",meminfo.used);
        
        printf("\n");
	}//end of for loop for no device
	
	printf("\n");
	printf("----------------------------------------------------------------------------\n");



    result = nvmlShutdown();
    if (NVML_SUCCESS != result)
    printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
    printf("All done.\n");
    return 0;

Error:
    result = nvmlShutdown();
    if (NVML_SUCCESS != result)
        printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));

    printf("Press ENTER to continue...\n");
    getchar();
    return 1;
}


	
