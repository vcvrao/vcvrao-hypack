/****************************************************************************

                C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Objective   : program to solve a Matrix vector multiplication using block
                striped partioning on hybrid computing using Pthreads and CUDA

  Input       : Matrix Rows, Matrix Columns, Vector Size and Process 0
                intializes the Matrix and the vector.

  Output      : Process 0 prints the resultant vector.

  Necessary      Number of rows of matrix should be greater than number of
  Conditons   :  processes and perfectly divisible by number of processes.
                 Matrix columns  should be equal to vector size.

  Created     : August-2013

  E-mail      : hpcfte@cdac.in

/****************************************************************************

#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<cuda.h>
//----------------------------------------------------------------------------------------------------------------------------

#define MAXTHREADS 8
#define BLOCKSIZE 8

//-------------------------------------------------------------------------------------------------------------------------------

pthread_mutex_t MutexResultVector = PTHREAD_MUTEX_INITIALIZER;
int CheckDevice(int );

//-------------------------------------------------------------------------------------------------------------------------------

float *MatrixA, *VectorB, *ResultVector;
int RowsNo, ColsNo,VectorSize, ThreadPart;

//------------------------------------------------------------------------------------------------------------------------------

void IntializingMatrixVectors(float **, float **, float **, int , int , int );

//------------------------------------------------------------------------------------------------------------------------------

//Pragma routine to report the detail of cuda error
#define CUDA_SAFE_CALL(call)                                                         \
            do{                                                                      \
                 cudaError_t err = call;                                             \
                 if(err != cudaSuccess)                                              \
                 {                                                                   \
                        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                         __FILE__, __LINE__, cudaGetErrorString( err) );             \
                         exit(1);                                                    \
                 }                                                                   \
               } while (0)                                                           \
//--------------------------------------------------------------------------------------------------------------------------------

//Kernel that performs Matrix Vector Multiplication
__global__ void MatrixVectorMultiplication(float *Matrix,float *Vector,float *Solution,int VectorLength, int Size, int ThreadDim)
{
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;
        int ThreadIndex = (ThreadDim * tidx) + tidy;
        int MaxNumThread = ThreadDim * ThreadDim;
        int count,ThreadColumnIndex,pass = 0 ;
        float TempResult = 0.0f;
        while( (ThreadColumnIndex = (ThreadIndex + MaxNumThread * pass))  <     Size )
        {
                TempResult = 0.0f;
                for( count = 0; count < VectorLength; count++)
                        TempResult +=  Matrix[ThreadColumnIndex*VectorLength+count] * Vector[count];
                Solution[ThreadColumnIndex] = TempResult;
                pass++;
        }
        __syncthreads();
}//End of Matrix Vector Multiplication Device Function
//----------------------------------------------------------------------------------------------------------------------------------

void* ThreadWork(int ThreadId)
{
	int Counter, Index = 0, DeviceStatus;
	//int ThreadId = *((int *)Id);
	int IndexValue, IndexCol;
	float *MyMatrixA, *MyResultVector;
	float *DeviceMyMatrixA, *DeviceVectorB, *DeviceMyResultVector;

	//Allocating memory on the host by the thread
	MyMatrixA = (float *)malloc(ThreadPart * ColsNo * sizeof(float));
	if(MyMatrixA == NULL)
	{
		printf("Memory is not available for the variable MyMatrixA \n");
		exit(-1);
	}
	MyResultVector = (float *)malloc(ThreadPart * sizeof(float));
	if(MyResultVector == NULL)
	{
		printf("Memory is not available for the variable MyResultVector \n");
		exit(-1);
	}

	//Filling the elements of MyMatrixA from the MatrixA
	for(Counter = ( (ThreadId -1) * ThreadPart * ColsNo ); Counter <= ((ThreadId * ThreadPart * ColsNo) - 1); Counter++)
		MyMatrixA[Index++] = MatrixA[Counter];

	DeviceStatus = CheckDevice(ThreadId);
	
	if(DeviceStatus == 0)
        {
                printf("Thread with Id %d doing partial product of Matrix and  vector on CPU \n",ThreadId);
                for(Index = 0 ; Index < ThreadPart ; Index++)
                {
                        MyResultVector[Index] = 0;
                        IndexValue = Index * ColsNo;
                        for(IndexCol = 0; IndexCol < ColsNo; IndexCol++)
                                MyResultVector[Index] += (MyMatrixA[IndexValue++] * VectorB[IndexCol]);
                }
        }
	else
	{
		//Defining Thread Grid and Thread Block
        	dim3 DimGrid(1, 1);
       		dim3 DimBlock(BLOCKSIZE, BLOCKSIZE);

		//Allocating the Memory on the device memory
        	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMyMatrixA, ThreadPart * ColsNo * sizeof(float) ) );
        	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMyResultVector, ThreadPart * sizeof(float) ) );
        	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceVectorB, VectorSize * sizeof(float) ) );

		//Copying the data from host to device
        	cudaMemcpy( (void *)DeviceMyMatrixA, (void *)MyMatrixA, ThreadPart * ColsNo * sizeof(float), cudaMemcpyHostToDevice );
        	cudaMemcpy( (void *)DeviceVectorB, (void *)VectorB, VectorSize * sizeof(float), cudaMemcpyHostToDevice );

		//Calling the kernel which performs Matrix Vector Product
        	MatrixVectorMultiplication<<<DimGrid, DimBlock>>>(DeviceMyMatrixA, DeviceVectorB, DeviceMyResultVector, ColsNo, ThreadPart, BLOCKSIZE);
	
		//Copying the value of patial result vector from device to host
        	cudaMemcpy( (void *)MyResultVector, (void *)DeviceMyResultVector, ThreadPart * sizeof(float), cudaMemcpyDeviceToHost );
	}

	pthread_mutex_lock(&MutexResultVector);

	//Filling the final result vector form the My Result Vector of each thread
	Index = (ThreadId - 1) * ThreadPart;
	for(Counter = 0; Counter < ThreadPart; Counter++)
		ResultVector[Index++] = MyResultVector[Counter];
	
	pthread_mutex_unlock(&MutexResultVector);
	
	//Freeing the host memory
	free(MyMatrixA);
	free(MyResultVector);

	//Freeing the device memory
	cudaFree(DeviceMyMatrixA);
	cudaFree(DeviceMyResultVector);
	cudaFree(DeviceVectorB);
	
	pthread_exit((void *) NULL);
}//End of Thread work
//--------------------------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
	pthread_t *Threads;
	pthread_attr_t ThreadAttribute;
	int  NumberOfThreads, ThreadCount, ThreadStatus;
	int Index;
	
	//Checking if valid number of arguements have been passed
	if(argc != 5)
	{
		printf(" Usage:<./Program Name><Number of Rows><Number of Columns><Vector Size><Number of threads> \n");
		exit(-1);
	}

	//Getting values from the program arguements
	RowsNo = atoi(argv[1]);
	ColsNo = atoi(argv[2]);
	VectorSize = atoi(argv[3]);
	NumberOfThreads = atoi(argv[4]);

	//Checking if Columns size is not equal to vector size 
	if(ColsNo != VectorSize)
	{
		printf("In matrix vector multiplication, Column size of matrix should be equal to Vector size \n");
		exit(-1);
	}

	//Checking if given thread number is greater than maximum number of threads
	if(NumberOfThreads > MAXTHREADS)
	{
		printf("The Threads that are to be created should be less than or equal to  %d \n", MAXTHREADS);
		exit(-1);
	}
	
	//Checking if rows can be shared equally by all the threads 
	if(RowsNo % NumberOfThreads != 0)
	{
		printf("The matrix can not be assesed by all the threads with the same amount of elements \n");
		exit(-1);
	}

	//Intializing the Matrix and the Vectors
	IntializingMatrixVectors(&MatrixA, &VectorB, &ResultVector, RowsNo, ColsNo, VectorSize);

	pthread_attr_init(&ThreadAttribute);

	//Allocating memory for the Threads
	Threads = (pthread_t *)malloc(NumberOfThreads * sizeof(pthread_t));
	if(Threads == NULL)
	{
		printf("Memory is not available for the variable Threads \n");
		exit(-1);
	}

	ThreadPart = RowsNo / NumberOfThreads;

	//Creating the Threads
	for(ThreadCount = 0; ThreadCount < NumberOfThreads; ThreadCount++)
	{
	  ThreadStatus = pthread_create(&Threads[ThreadCount], &ThreadAttribute, (void *(*) (void *)) ThreadWork, (void *)(ThreadCount+1));
		if(ThreadStatus)
		{
			printf("Error in creating the thread and the return status is %d \n",ThreadStatus);
			exit(-1);
		}
	}

	//Joining the threads
	for(ThreadCount = 0; ThreadCount < NumberOfThreads; ThreadCount++)
	{
		ThreadStatus = pthread_join(Threads[ThreadCount], NULL);
		if(ThreadStatus)
		{
			printf("Error in joining the threadsand the return status is %d \n",ThreadStatus);
			exit(-1);
		}
	}

	//Printing the resultant vector
	printf("The result vector with the size %d  is \n", RowsNo);
	for(Index = 0; Index < RowsNo; Index++)
		printf("%f \n", ResultVector[Index]);  
	
	ThreadStatus = pthread_attr_destroy(&ThreadAttribute);
	if(ThreadStatus)
	{
		printf("Error in pthread_attr_destroy and the return status is %d \n", ThreadStatus);
		exit(-1);
	}
	
	//Freeing the host memory
	free(MatrixA);
	free(VectorB);
	free(ResultVector);

	return(0);

}//End of main
//--------------------------------------------------------------------------------------------------------------------------------------------------

void IntializingMatrixVectors(float **MatrixA, float **VectorB, float **ResultVector, int RowsNo, int ColsNo, int VectorSize)
{
        float *TempMatrixA, *TempVectorB, *TempResultVector;
        int Index;

        //Allocating memory on the host
        TempMatrixA = (float *)malloc(RowsNo * ColsNo * sizeof(float));
        if(TempMatrixA == NULL)
        {
		printf("Memory is not available for the variable TempMatrixA \n");
		exit(-1);
	}
        TempVectorB = (float *)malloc(VectorSize * sizeof(float));
        if(TempVectorB == NULL)
	{
		printf("Memory is not available for the variable TempVectorB \n");
		exit(-1);
	}
        TempResultVector = (float *)malloc(RowsNo * sizeof(float));
        if(TempResultVector == NULL)
        {
		printf("Memory is not available for the variable TempResultVector \n");
		exit(-1);
	}

        //Intializing the Matrix and the Vectors
        for(Index = 0; Index < RowsNo * ColsNo; Index++)
                TempMatrixA[Index] = 1.0f;
        for(Index = 0; Index < VectorSize; Index++)
                TempVectorB[Index] = 1.0f;
        for(Index = 0; Index < RowsNo; Index++)
                TempResultVector[Index] = 0.0f;

        *MatrixA = TempMatrixA;
        *VectorB = TempVectorB;
        *ResultVector = TempResultVector;
}//End of the function
//-------------------------------------------------------------------------------------------------------------------------------------------------------
int CheckDevice(int ThreadId)
{
        int DeviceCount, Device;
        struct cudaDeviceProp Properties;

        cudaGetDeviceCount(&DeviceCount);
        if(DeviceCount >= 1)
        {
                cudaGetDevice(&Device);
                cudaGetDeviceProperties(&Properties, Device);
                printf("Thread with  Id %d has the Device by name %s and computation is done on this device \n",ThreadId, Properties.name);
        }

        return(DeviceCount);

}//End of function
//--------------------------------------------------------------------------------------------------------------

