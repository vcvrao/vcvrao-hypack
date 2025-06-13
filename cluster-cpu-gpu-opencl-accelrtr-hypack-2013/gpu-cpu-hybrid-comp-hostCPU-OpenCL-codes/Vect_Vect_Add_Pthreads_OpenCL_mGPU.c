/***************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

Example : Vect_Vect_Add_Pthreads_OpenCL_mGPU.c

Objective : Example program to demonstrate mixed mode programming using
            opencl and pthreads for Vector Addition.

Description : Here we are performing vector addition using opencl-pthread
              mixed model. We are taking vector length fro the user.
              According to number of devices available program will choose
              number of threads and launch on separate device.

Condition : Vector length should be divisible by number of devices available.
            
Input : Vector Length

Output : Resultant Vector (along with Input Vectors )

Created     : August-2013

E-mail      : hpcfte@cdac.in     

*****************************************************************************/

#include<CL/cl.h>
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<pthread.h>
#include<string.h>

//-----------------------------------------------Kernel Function Starts--------------------------------------------------------------

// string containing kernel source
const char *programSource =
"__kernel void vectorAddition(__global int* input_vecA, __global int* input_vecB,__global int* result_vec, __global int* vecLen)\n"
"{\n"
"       int threadGid = get_global_id(0);\n"
"       int numThread = get_global_size(0);\n"
"	int vecCount;\n"
"       if( threadGid < (*vecLen))\n"
"       {\n"
"         for( vecCount = threadGid; vecCount < (*vecLen); vecCount = vecCount + numThread)\n"
"         {\n"
"		result_vec[vecCount]  = input_vecA[vecCount] + input_vecB[vecCount];\n"
"         }\n"
"       }\n"
"}\n";

//--------------------------------------------------Kernel Function Ends--------------------------------------------------------------


//-------------------------------------------------- Declaration Part Starts-------------------------------------------------------

cl_int status = CL_SUCCESS;    // integer value used to check whether operation is successfull or not

// variables declared for platform information
cl_uint maxNumofPlatforms;
cl_uint numOfPlatformsAvailable;
cl_platform_id *platforms;
char platformName[200];
cl_int platformId = -1;

// variables declared for device information
cl_uint numOfDevicesAvailable;
cl_device_id *devices;
cl_device_type deviceType;
char deviceBuffer[100];
char **deviceName;

// context related variables
cl_context_properties contextProp[3];
cl_context context;

cl_command_queue *cmdQueue;   // command Q variable
cl_program progObject;       // program object
cl_kernel *myKernelVecAdd;   // kernel object


size_t global[1];  // array to define global work size

void* threadWorkVecAddition(int); // thread function
void check_status(char*);       // function to check whether operation is successfull or not
void getPlatformDeviceInfo();
int checkResultCPUGPU(int*,int*,int*,int); // check correctness of GPU vs CPU

// variables for vector addition on host
int *vectorA , *vectorB , *resultVector;
int vectorLength , numThreads , threadPart;


//-------------------------------------------------- Declaration Part Ends---------------------------------------------------


//-------------------------------------------------- Main Starts---------------------------------------------------
int main(int argc , char *argv[])
{
	pthread_t *threads;
	int threadCount , threadStatus , vecCount , chkresult ;

	if(argc != 2)
	{
		printf("\n Usage :<executable> <vectorLength>");
		exit(0);
	}

	vectorLength = atoi(argv[1]);
		

	getPlatformDeviceInfo();           // call the function to get platform and device information


	 if((vectorLength % numThreads) != 0)           // check input condition
         {
                printf("\n Vector Length should be divisible by numOfDevicesAvailable......\n\n");
                exit(0);
         }

	 threadPart = vectorLength / numThreads ;    // define chunk size of each thread

	/*  Create Context to associate devices
	    context properties list - must be terminated with 0 */

        contextProp[0] = CL_CONTEXT_PLATFORM;                          // prop name
        contextProp[1] = (cl_context_properties)platforms[platformId];          // prop value
        contextProp[2] = 0;                                            // must be terminated with 0

        context = clCreateContext(contextProp , numOfDevicesAvailable , devices ,NULL , NULL , &status);
        check_status("clCreateContext");

        //  command Queue memory allocation according to num of device available
	assert( (cmdQueue=(cl_command_queue *) malloc (sizeof(cl_command_queue) * numOfDevicesAvailable))) ;
        printf("\n"); 

	 //  Create program object
        progObject = clCreateProgramWithSource(context , 1 , &programSource , NULL , &status);
        check_status("clCreateProgramWithSource");

        // Build Program Executables
        status = clBuildProgram(progObject,numOfDevicesAvailable ,devices , NULL,NULL,NULL);
        check_status("clBuildProgram");

        //  kernel Object memory allocation according to num of device available
	assert((myKernelVecAdd = (cl_kernel *) malloc (sizeof(cl_kernel) * numOfDevicesAvailable)));
	
        // Memory allocation and data filling on host

	assert(vectorA = (int *)malloc(vectorLength * sizeof(int)));
	assert(vectorB = (int *)malloc(vectorLength * sizeof(int)));
	assert(resultVector = (int *)malloc(vectorLength * sizeof(int)));


	for( vecCount = 0; vecCount< vectorLength ; vecCount++ )
	{
		vectorA[vecCount] = (int) rand()%10;
		vectorB[vecCount] = (int) rand()%10;
		resultVector[vecCount] = 0;
	}


	global[0] = threadPart;	     //  global work size for each kernel
	

	assert(threads = (pthread_t *)malloc(numThreads * sizeof(pthread_t)));   // allocate memory for number of threads

	// Call thread function
	for(threadCount = 0 ; threadCount < numThreads ; threadCount++)
	{
	      threadStatus = pthread_create(&threads[threadCount], NULL,  (void *(*) (void *))threadWorkVecAddition, (void *)(threadCount));
	      if(threadStatus)
		{
			printf("Error in creating the thread and the return status is %d \n",threadStatus);
			exit(-1);
		}
	}


	// join threads with main thread
	for(threadCount = 0 ; threadCount < numThreads ; threadCount++)
        {
              threadStatus = pthread_join(threads[threadCount], NULL);
              if(threadStatus)
                {
                        printf("Error in joining the threads and the return status is %d \n",threadStatus);
                        exit(-1);
                }
        }

	chkresult = checkResultCPUGPU(vectorA,vectorB,resultVector,vectorLength);	
	if(chkresult)
	{
		printf("\n Uncorrect result :CPU and GPU results are not same.....Aborting... \n");
		exit(1);
	}

	printf("\n\n\n Successfull operation...CPU and GPU results are same....!!\n\n");
	 // Print Input and resultant vectors
	 /*printf("\n Input VectorA \n");
         for( vecCount = 0; vecCount< vectorLength ; vecCount++ )
         {
                printf("%d\t", vectorA[vecCount]);
         }
	 printf("\n");

	 printf("\n Input VectorB \n");
         for( vecCount = 0; vecCount< vectorLength ; vecCount++ )
         {
                printf("%d\t", vectorB[vecCount]);
         }
	 printf("\n");

	printf("\n Result Vector \n");
	 for( vecCount = 0; vecCount< vectorLength ; vecCount++ )
         {
		printf("%d\t", resultVector[vecCount]);
	 }*/

	 printf("\n");
	 
	// Cleanup 
	status = clReleaseProgram(progObject);
        check_status(" clReleaseProgram progObject");

	status = clReleaseContext(context);
        check_status("clReleaseContext");

	 free(vectorA);
	 free(vectorB);
	 free(resultVector);
	
return 0;
}


//-------------------------------------------------- Main Ends---------------------------------------------------



//-------------------------------------------------- Thread Function Starts---------------------------------------------------

// thread function will launch the kernel according to threadId
void* threadWorkVecAddition(int threadId)
{
         int iStrt , iEnd ,i ;     

	 cl_mem devcVectorA , devcVectorB , devcResultVector ,devcVecLen;
         // Create command Queue
         cmdQueue[threadId] = clCreateCommandQueue( context, devices[threadId], 0, &status);
         check_status("clCreateCommandQueue");
         printf("\n\t  Command Queue has been created for device = %s ",deviceName[threadId]);
	 printf("\n");
	 
         // Create kernel object
         myKernelVecAdd[threadId] = clCreateKernel ( progObject, "vectorAddition", &status);
         check_status(" clCreateKernel ");
         printf("\n\t Kernel Object has been created for device = %s ",deviceName[threadId]);
	printf("\n");

         // chunk size range for each thread
         iStrt = threadId * threadPart ;
         iEnd  = iStrt + (threadPart-1);
         printf(" \n\n\t Chunk size for device %s is from %d to %d", deviceName[threadId],iStrt,iEnd);

         // create buffer objects
         devcVectorA = clCreateBuffer ( context, CL_MEM_READ_ONLY , threadPart  * sizeof(int),NULL, &status);
         check_status("devcVectorA");
         devcVectorB = clCreateBuffer ( context, CL_MEM_READ_ONLY , threadPart  * sizeof(int),NULL, &status);
         check_status("devcVectorB");
         devcResultVector = clCreateBuffer ( context, CL_MEM_WRITE_ONLY , threadPart  * sizeof(int),NULL, &status);
         check_status("devcResultVector");
         devcVecLen = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(cl_int), (void*) &threadPart, &status);
         check_status(" create buffer object for devcVecLen");

         // copy data from host to device memory
         status = clEnqueueWriteBuffer (cmdQueue[threadId], devcVectorA, CL_TRUE, 0, threadPart * sizeof(float), &vectorA[threadId * threadPart], 0, 0 , 0);
         check_status("clEnqueueWriteBuffer for VectorA");
         status = clEnqueueWriteBuffer (cmdQueue[threadId], devcVectorB, CL_TRUE, 0, threadPart * sizeof(float), &vectorB[threadId * threadPart], 0, 0 , 0);
         check_status("clEnqueueWriteBuffer for VectorB");

	  // set kernel arguments
         status = clSetKernelArg(myKernelVecAdd[threadId],0,sizeof(cl_mem),(void *) &devcVectorA);
         check_status("clSetKernelArg 0");
         status = clSetKernelArg(myKernelVecAdd[threadId],1,sizeof(cl_mem),(void *) &devcVectorB);
         check_status("clSetKernelArg 1");
         status = clSetKernelArg(myKernelVecAdd[threadId],2,sizeof(cl_mem),(void *) &devcResultVector);
         check_status("clSetKernelArg 2");
         status = clSetKernelArg(myKernelVecAdd[threadId],3,sizeof(cl_mem),(void *) &devcVecLen);
         check_status("clSetKernelArg 3");

         // enqueue kernel for execution in command queue
         status = clEnqueueNDRangeKernel(cmdQueue[threadId] , myKernelVecAdd[threadId] , 1 , NULL , global , NULL , 0 , NULL , NULL);
         check_status("clEnqueueNDRangeKernel");

         // read result from device into host memory
         status = clEnqueueReadBuffer(cmdQueue[threadId] , devcResultVector , CL_TRUE , 0 , threadPart * sizeof(cl_int) , &resultVector[threadId*threadPart] ,0 ,NULL,NULL);
        check_status("clEnqueueReadBuffer");

	//----------- uncomment following if you want to see chunk of calcualtion done by this thread--------------------------------
	/* printf("\n-------------------------------------------------------------------------------------------------------\n");
	 printf("\n Device = %s has done following part of calculation from %d to %d \n",deviceName[threadId] ,iStrt ,iEnd);
	 printf("\n Input Vector A \n");
         for(i=iStrt; i <= iEnd ; i++)
	 {
		printf("%d\t",vectorA[i]);
	 }
	 printf("\n");
	
	 printf(" \nInput Vector B \n");
         for(i=iStrt; i <= iEnd ; i++)
	 {
		printf("%d\t",vectorB[i]);
	 }
	 printf("\n");

	 printf(" \nResult vector for this chunk \n");
         for(i=iStrt; i <= iEnd ; i++)
         {
                printf("%d\t",resultVector[i]);
         }
	 printf("\n");

	 printf("\n-------------------------------------------------------------------------------------------------------\n");*/

        //----------------------------------------------uncomment upto here-----------------------------------------------------------

         // wait until all commands complete execution
         status = clFinish(cmdQueue[threadId]);
         check_status("clFinish");
        
          // CleanUp    
          clReleaseMemObject(devcVectorA);
          check_status("clReleaseMemObject devcMyVectorA");
          clReleaseMemObject(devcVectorB);
          check_status("clReleaseMemObject devcMyVectorB");
          clReleaseMemObject(devcResultVector);
          check_status("clReleaseMemObject devcMyResultVector");
          clReleaseMemObject(devcVecLen);
          check_status("clReleaseMemObject devcVecLen");

          clReleaseKernel(myKernelVecAdd[threadId]);
          check_status("clReleaseKernel");
          status = clReleaseCommandQueue(cmdQueue[threadId]);
          check_status("clReleaseCommandQueue");

	   pthread_exit((void *) NULL);
}


//-------------------------------------------------- Thread Function Ends---------------------------------------------------


//-------------------------------------------------- check_status Function Starts---------------------------------------------------

// Function to check correctness of OpenCL APIs
void check_status(char* op)
{
        if(status != CL_SUCCESS)
        {
                printf("\n Operation %s is not sucessfull.....\n",op);
                exit(1);
        }
}

//-------------------------------------------------- check_status Function Ends---------------------------------------------------


//-------------------------------------------------- gettformDeviceInfo() Function Starts---------------------------------------------

void getPlatformDeviceInfo()
{
	int i , j ;

	// 1 .query host system for platform information
	status = clGetPlatformIDs(0 , 0 , &numOfPlatformsAvailable);
        check_status("clGetPlatformIDs");

        assert(platforms = (cl_platform_id*)malloc(numOfPlatformsAvailable * sizeof(cl_platform_id)));
        status = clGetPlatformIDs(numOfPlatformsAvailable , platforms , NULL);
        check_status("clGetPlatformIDs");

        printf("\n Total number of platforms available are %d \n",numOfPlatformsAvailable);

        for(i = 0 ; i < numOfPlatformsAvailable ; i++)
        {
                status = clGetPlatformInfo(platforms[i] , CL_PLATFORM_NAME , sizeof(platformName) , &platformName , NULL);
                check_status("clGetPlatformInfo for CL_PLATFORM_NAME");
                platformId = i;
                printf("\n PLATFORM_NAME = %s \n",platformName);        
        }

	for(i = 0 ; i < numOfPlatformsAvailable ; i++)
        {
                status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU , 0 , 0 , &numOfDevicesAvailable);
                if(status != CL_SUCCESS || numOfDevicesAvailable == 0)
                {
                       status = clGetPlatformInfo(platforms[i] , CL_PLATFORM_NAME , sizeof(platformName) , &platformName , NULL);
                        printf("\n------------------------------------------------------------------\n");
                        printf("\nNo GPU device found for the %s\n",platformName);
                        printf("\n------------------------------------------------------------------\n");
                        continue;
                }

                else
                {
                         assert(devices = (cl_device_id*)malloc(numOfDevicesAvailable * sizeof(cl_device_id)));
                         status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU , numOfDevicesAvailable , devices , 0);
                         platformId = i;
                }

        }



        //2.query host system for device information

              assert( (deviceName = (char **) malloc( sizeof(char *) * numOfDevicesAvailable)) != NULL);
              for ( i = 0 ; i < numOfDevicesAvailable ; i++)
              { 
                // Get devices Name 
                status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(deviceBuffer), &deviceBuffer, NULL);
                check_status("clGetDeviceInfo ");
                assert( (deviceName[i] = (char *) malloc( sizeof(char ) * sizeof(deviceBuffer))) != NULL);
                strcpy(deviceName[i],deviceBuffer);
                printf("\n OpenCL Device Name \t= %s ",deviceBuffer);

             }
                
            printf("\n");

         numThreads = numOfDevicesAvailable ;

}
//-------------------------------------------------- gettformDeviceInfo() Function Ends---------------------------------------------



//-------------------------------------------------- checkResultCPUGPU Function Starts---------------------------------------------
int checkResultCPUGPU(int* vectorA,int* vectorB,int* resultVector,int vectorLength)
{
	int i,err = 0;
	for(i=0;i<vectorLength;i++)
	{
		if( (vectorA[i]+vectorB[i]) != resultVector[i])
		{
			err = 1 ;
			break;
		}
	}

	return err;
}
//-------------------------------------------------- checkResultCPUGPU Function Ends---------------------------------------------
