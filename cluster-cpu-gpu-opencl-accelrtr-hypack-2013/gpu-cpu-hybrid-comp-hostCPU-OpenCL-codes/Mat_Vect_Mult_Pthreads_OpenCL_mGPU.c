/***************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

Example : Mat_Vect_Mult_Pthreads_OpenCL_mGPU.c

Objective : Example program to demonstrate mixed mode programming using
            opencl and pthreads for Matrix-Vector Multiplication.

Description : Here we are performing  Matrix-Vector Multiplication using opencl-pthread
              mixed model. We are taking  number of rows and number of columns/ 
	      vectorLenght from the user.
              According to number of devices available program will choose
              number of threads and launch on separate device.

Condition : Number of rows should be divisible by number of devices available.


Input : Number of Rows , Vector Length / Number of Columns

Output : Resultant Vector (along with Input Matrix and Vector )

Created   : August-2013

E-mail    : hpcfte@cdac.in     

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
"__kernel void matrixvectorMult(__global int* input_mat, __global int* input_vec,__global int* result_vec, __global int* vecLen , __global int* numRows)\n"
"{\n"
"       int threadGid = get_global_id(0);\n"
"       int numThread = get_global_size(0);\n"
"       int tempResult = 0; \n"
"       if( threadGid < (*numRows))\n"
"       {\n"
"         for( int rowCount = threadGid; rowCount < (*numRows); rowCount = rowCount + 1)\n"
"         {\n"
"		tempResult = 0;\n"
"               for(int colCount = 0; colCount < (*vecLen); colCount++ )\n"
"               {\n"
"                       tempResult = tempResult + input_vec[colCount] * input_mat[rowCount * (*vecLen) + colCount];\n"
"               }\n"
"               result_vec[rowCount] = tempResult; \n"
"         }\n"
"        }\n"
"}\n";

//--------------------------------------------------Kernel Function Ends--------------------------------------------------------------


//-------------------------------------------------- Declaration Part Starts-------------------------------------------------------

cl_int status = CL_SUCCESS;    // integer value used to check whether operation is successfull or not

// variables declared for platform information
cl_uint maxNumofPlatforms;
cl_uint numOfPlatformsAvailable;
cl_platform_id *platforms;
char platformName[200];
cl_int platformId;

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
cl_kernel *myKernelMatVectMult;   // kernel object


size_t global[1];  // array to define global work size

void* threadWorkMatVecMultiplication(int); // thread function
void check_status(char*);                  // function to check whether operation is successfull or not
void getPlatformDeviceInfo();              // function to get platform and device information
int checkResultCPUGPU(int*,int*,int,int); // check correctness of GPU vs CPU

// variables for vector addition on host
cl_int *hst_inputMat , *hst_inputVec;
cl_int *hst_resultVec;
cl_int numRows,numCols,vecLength;

int  numThreads , threadPart;


//-------------------------------------------------- Declaration Part Ends---------------------------------------------------


//-------------------------------------------------- Main Starts---------------------------------------------------
int main(int argc , char *argv[])
{
	pthread_t *threads;
	int threadCount , threadStatus , chkresult ;
	int i ,j;
        size_t rowCount ,colCount ,vecCount ;

        if(argc!=3)
        {
                printf("\n Usage :<executable> <numRows> <numCols - vecLength>\n");
                exit(0);
        }

        numRows = atoi(argv[1]);
        numCols = vecLength = atoi(argv[2]);


	getPlatformDeviceInfo();           // call the function to get platform and device information


	 if((numRows % numThreads) != 0)           // check input condition
         {
                printf("\n Number of Rows should be divisible by  numOfDevicesAvailable......\n\n");
                exit(0);
         }

	 threadPart = numRows / numThreads ;    // define chunk size of each thread


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
	assert((myKernelMatVectMult = (cl_kernel *) malloc (sizeof(cl_kernel) * numOfDevicesAvailable)));
	
        // Memory allocation and data filling on host

	assert(hst_inputMat =  (cl_int*)malloc(numRows*numCols*sizeof(cl_int)));
        assert(hst_inputVec =  (cl_int*)malloc(vecLength*sizeof(cl_int)));
        assert(hst_resultVec = (cl_int*)malloc(numRows*sizeof(cl_int)));


	for(rowCount=0; rowCount < numRows; rowCount++)
           for(colCount=0; colCount < numCols; colCount++)
               hst_inputMat[rowCount * numCols + colCount] = (cl_int)rand()%10;

        for(vecCount=0; vecCount < vecLength; vecCount++)
               hst_inputVec[vecCount] = (cl_int)rand()%10;

        for(colCount=0; colCount < numRows; colCount++)
               hst_resultVec[colCount] = 0;


	global[0] = threadPart;	     //  global work size for each kernel
	

	assert(threads = (pthread_t *)malloc(numThreads * sizeof(pthread_t)));   // allocate memory for number of threads

	// Call thread function
	for(threadCount = 0 ; threadCount < numThreads ; threadCount++)
	{
	      threadStatus = pthread_create(&threads[threadCount], NULL,  (void *(*) (void *))threadWorkMatVecMultiplication, (void *)(threadCount));
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
	

	// Check correctness of result
	chkresult = checkResultCPUGPU(hst_inputMat, hst_inputVec,numRows,vecLength);	
	if(chkresult)
	{
		printf("\n Uncorrect result :CPU and GPU results are not same.....Aborting... \n");
		exit(1);
	}

	printf("\n Successfull....CPU and GPU Results are same....!!\n\n");
	 // Print Input Matrix,Vector and resultant vector
	 /*printf("\nInput Matrix \n");
         for(rowCount = 0; rowCount < numRows ; rowCount++)
         {
                printf("\n");
                for(colCount = 0; colCount < numCols; colCount++)
                {
                        printf("%d\t", hst_inputMat[rowCount*numCols+colCount]);
                }
        }
         printf("\n");

         printf("\nInput Vec \n");
         for(vecCount = 0; vecCount < vecLength ; vecCount++)
         {
               printf("%d\t", hst_inputVec[vecCount]);
         }
         printf("\n");

         printf("\nResult Vector \n");
         for(vecCount = 0; vecCount < numRows ; vecCount++)
         {
               printf("%d\t", hst_resultVec[vecCount]);
         }*/
         printf("\n");

	 
	// Cleanup 
	status = clReleaseProgram(progObject);
        check_status(" clReleaseProgram progObject");

	status = clReleaseContext(context);
        check_status("clReleaseContext");

	free(hst_inputMat);
        free(hst_inputVec);
        free(hst_resultVec);
	
	return 0;
}


//-------------------------------------------------- Main Ends---------------------------------------------------



//-------------------------------------------------- Thread Function Starts---------------------------------------------------

// thread function will launch the kernel according to threadId
void* threadWorkMatVecMultiplication(int threadId)
{
         int iStrt , iEnd ,i ;     

	 cl_mem devc_inputMat , devc_inputVec;
	 cl_mem devc_resultVec;
	 cl_mem devc_vecLength , devc_numRows;


         // Create command Queue
         cmdQueue[threadId] = clCreateCommandQueue( context, devices[threadId], 0, &status);
         check_status("clCreateCommandQueue");
         printf("\n\t  Command Queue has been created for device = %s ",deviceName[threadId]);
	 printf("\n");
	 
         // Create kernel object
         myKernelMatVectMult[threadId] = clCreateKernel ( progObject, "matrixvectorMult", &status);
         check_status(" clCreateKernel ");
         printf("\n\t Kernel Object has been created for device = %s ",deviceName[threadId]);
	 printf("\n");

         // chunk size range for each thread
         iStrt = threadId * (threadPart*numCols) ;
         iEnd  = iStrt + ((threadPart*numCols)-1);
         printf(" \n\n\t Chunk size for device %s is from %d to %d", deviceName[threadId],iStrt,iEnd);



         // create buffer objects
         devc_inputMat = clCreateBuffer ( context, CL_MEM_READ_ONLY , numCols * threadPart  * sizeof(int),NULL, &status);
         check_status("devc_inputMat");
         devc_inputVec = clCreateBuffer ( context, CL_MEM_READ_ONLY , vecLength  * sizeof(int),NULL, &status);
         check_status("devc_inputVec");
         devc_resultVec = clCreateBuffer ( context, CL_MEM_WRITE_ONLY , threadPart  * sizeof(int),NULL, &status);
         check_status("devc_resultVec");
         devc_numRows = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(cl_int), (void*) &threadPart, &status);
         check_status(" create buffer object for devc_numRows");
         devc_vecLength = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(cl_int), (void*) &vecLength, &status);
         check_status(" create buffer object for devc_vecLength");

         // copy data from host to device memory
         status = clEnqueueWriteBuffer (cmdQueue[threadId], devc_inputMat, CL_TRUE, 0, numCols * threadPart * sizeof(float), &hst_inputMat[threadId * threadPart * numCols], 0, 0 , 0);
         check_status("clEnqueueWriteBuffer for VectorA");
         status = clEnqueueWriteBuffer (cmdQueue[threadId], devc_inputVec, CL_TRUE, 0, vecLength * sizeof(float), hst_inputVec, 0, 0 , 0);
         check_status("clEnqueueWriteBuffer for VectorB");

	  // set kernel arguments
         status = clSetKernelArg( myKernelMatVectMult[threadId],0,sizeof(cl_mem),(void *) &devc_inputMat);
         check_status("clSetKernelArg 0");
         status = clSetKernelArg( myKernelMatVectMult[threadId],1,sizeof(cl_mem),(void *) &devc_inputVec);
         check_status("clSetKernelArg 1");
         status = clSetKernelArg( myKernelMatVectMult[threadId],2,sizeof(cl_mem),(void *) &devc_resultVec);
         check_status("clSetKernelArg 2");
         status = clSetKernelArg( myKernelMatVectMult[threadId],3,sizeof(cl_mem),(void *) &devc_vecLength);
         check_status("clSetKernelArg 3");
	 status = clSetKernelArg( myKernelMatVectMult[threadId],4,sizeof(cl_mem),(void *) &devc_numRows);
         check_status("clSetKernelArg 4");

         // enqueue kernel for execution in command queue
         status = clEnqueueNDRangeKernel(cmdQueue[threadId] , myKernelMatVectMult[threadId] , 1 , NULL , global , NULL , 0 , NULL , NULL);
         check_status("clEnqueueNDRangeKernel");

         // read result from device into host memory
         status = clEnqueueReadBuffer(cmdQueue[threadId] , devc_resultVec , CL_TRUE , 0 , threadPart * sizeof(cl_int) , &hst_resultVec[threadId*threadPart] ,0 ,NULL,NULL);
        check_status("clEnqueueReadBuffer");


         // wait until all commands complete execution
          status=clFinish(cmdQueue[threadId]);
         check_status("clFinish");

        
          // CleanUp    
          clReleaseMemObject(devc_inputMat);
          check_status("clReleaseMemObject devc_inputMat");
 	  clReleaseMemObject(devc_inputVec);
          check_status("clReleaseMemObject devc_inputVec");
          clReleaseMemObject(devc_resultVec);
          check_status("clReleaseMemObject devc_resultVec");
          clReleaseMemObject(devc_vecLength);
          check_status("clReleaseMemObject devc_vecLength");
	  clReleaseMemObject(devc_numRows);
          check_status("clReleaseMemObject devc_numRows");

          clReleaseKernel(myKernelMatVectMult[threadId]);
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
                printf("\n GPU Device %d Name \t= %s ",i,deviceBuffer);

             }
                
            printf("\n");

         numThreads = numOfDevicesAvailable ;

}
//-------------------------------------------------- gettformDeviceInfo() Function Ends---------------------------------------------



//-------------------------------------------------- checkResultCPUGPU Function Starts---------------------------------------------
int checkResultCPUGPU(int* input_mat,int* input_vec,int numRows , int vecLength)
{
	int *result_vec;
	int rowCount,colCount ,err = 0 , tempResult ;
	assert(result_vec = (int *)malloc(sizeof(int)* numRows));
	
	for(  rowCount = 0; rowCount < numRows; rowCount++)
         {
		tempResult = 0;
               for(colCount = 0; colCount < vecLength; colCount++ )
               {
                       tempResult = tempResult + input_vec[colCount] * input_mat[(rowCount * vecLength) + colCount];
               }
               result_vec[rowCount] = tempResult; 
        }

	/*printf("\nCPU result vector\n");

	for(rowCount=0;rowCount < numRows; rowCount++)
	{
		printf("%d\t",result_vec[rowCount]);
	}

	printf("\nGPU result vector\n");

        for(rowCount=0;rowCount < numRows; rowCount++)
        {
                printf("%d\t",hst_resultVec[rowCount]);
        }*/
	
	printf("\n");

	 for(rowCount = 0; rowCount < numRows; rowCount++)
	 {
		if(result_vec[rowCount] == hst_resultVec[rowCount])
			err = 0;
		else
		{
			err = 1;
			free(result_vec);
		 	break;
		}
	 }

	//free(result_vec);
	return err;
}
//-------------------------------------------------- checkResultCPUGPU Function Ends---------------------------------------------
