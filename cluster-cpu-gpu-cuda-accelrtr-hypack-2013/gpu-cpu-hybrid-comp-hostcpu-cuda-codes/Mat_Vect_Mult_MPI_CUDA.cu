/****************************************************************************

                C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Objective   : program to solve a Matrix vector multiplication using block
                striped partioning on hybrid computing using MPI and CUDA

  Input       : Matrix Rows, Matrix Columns, Vector Size and Process 0
                intializes the Matrix and the vector.

  Output      : Process 0 prints the resultant vector.

  Necessary      Number of rows of matrix should be greater than number of
  Conditons   :  processes and perfectly divisible by number of processes.
                 Matrix columns  should be equal to vector size.

  Created     : August-2013

  E-mail      : hpcfte@cdac.in

****************************************************************************/

#include<stdio.h>
#include<cuda.h>
#include<mpi.h>
//------------------------------------------------------------------------------------------------------------------------------------------

#define BLOCKSIZE 16

//--------------------------------------------------------------------------------------------------------------------------------------------

int IntializingMatrixVectors(float **, float **, float **, int , int , int );
int CheckDevice(int );

//--------------------------------------------------------------------------------------------------------------------------------------------

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

//----------------------------------------------------------------------------------------------------------------------------------------

//Kernel that performs Matrix Vector Multiplication
__global__ void MatrixVectorMultiplication(float *Matrix,float *Vector,float *Solution,int VectorLength, int ScatterSize, int ThreadDim)
{
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;
        int ThreadIndex = (ThreadDim * tidx) + tidy;
        int MaxNumThread = ThreadDim * ThreadDim;
        //int VectLen = MatrixSize;
        int count,ThreadColumnIndex,pass = 0 ;
        float TempResult = 0.0f;
        while( (ThreadColumnIndex = (ThreadIndex + MaxNumThread * pass))  <     ScatterSize )
        {
                TempResult = 0.0f;
                for( count = 0; count < VectorLength; count++)
                        TempResult +=  Matrix[ThreadColumnIndex*VectorLength+count] * Vector[count];
                Solution[ThreadColumnIndex] = TempResult;
                pass++;
        }
        __syncthreads();
}//End of Matrix Vector Multiplication Device Function
//---------------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
	int MyRank, NumberOfProcessors;
	int Root = 0, Index, Status = 1;
	float *MatrixA, *VectorB, *ResultVector;
	float *MyMatrixA, *MyResultVector;
	float *DeviceMyMatrixA, *DeviceMyResultVector, *DeviceVectorB;
	int RowsNo, ColsNo, VectorSize, ScatterSize, IndexCol, IndexValue, DeviceStatus;

	//MPI Intialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
	MPI_Comm_size(MPI_COMM_WORLD, &NumberOfProcessors);

	//Checking if valid number of arguements have been passed
	if(argc != 4)
	{
		if(MyRank == Root)
			printf("Usage:<mpirun><-n><Number of processors><./Program Name><Number of Rows of Matrix><Number of Columns of Matrix><VectorSize> \n");
		MPI_Finalize();
		exit(-1);
	}

	//Assigning values to RowsNo, ColsNo, VectorSize from the arguements passed
	RowsNo = atoi( argv[1] );
	ColsNo = atoi( argv[2] );
	VectorSize = atoi( argv[3] );

	//Checking if columns is equal to vector size
	if(ColsNo != VectorSize)
	{
		if(MyRank == Root)
			printf("Entered wrong input, Number of columns of matrix should be equal to size of the vector \n");
		MPI_Finalize();
		exit(-1);
	}

	if(RowsNo < NumberOfProcessors)
	{
		if(MyRank == Root)
			printf("Given number of Rows of the matrix should be more than number of processors \n");
		MPI_Finalize();
		exit(-1);
	}

	//Checking if Matrix can be distributed evenly to all the nodes
	if(RowsNo % NumberOfProcessors != 0)
	{
		if(MyRank == Root)
			printf("The Rows of the matrix can not be distributed evenly among processors \n");
		MPI_Finalize();
		exit(-1);
	}

	//Root node intializes the Matrix, Vector and Result Vector
        if(MyRank == Root)
                Status = IntializingMatrixVectors(&MatrixA, &VectorB, &ResultVector, RowsNo, ColsNo, VectorSize);

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Bcast(&Status, 1, MPI_INT, Root, MPI_COMM_WORLD);

        //Checking if Status returned by the IntializingMatrixVectors
        if(Status == 0)
        {
                if(MyRank == Root)
                        printf("Memory is not available to allocate for the Variables \n");
                MPI_Finalize();
                exit(-1);
        }	
	
	//Allocating memory for the Vector by all nodes expect root node
	if(MyRank != Root)
		VectorB = (float *)malloc(VectorSize * sizeof(float));

	//Broad casting the Vector to all the nodes from root node
	MPI_Bcast(VectorB, VectorSize, MPI_FLOAT, Root, MPI_COMM_WORLD);

	//Calculating the Scatter size of the Matrix
	ScatterSize = RowsNo / NumberOfProcessors;

	//Allocating the memory on the host for the MyMatrixA and MyResultVector by all nodes
	MyMatrixA = (float *)malloc(ScatterSize * ColsNo * sizeof(float) );
	if(MyMatrixA == NULL)
		Status = 0;

	MyResultVector = (float *)malloc(ScatterSize * sizeof(float));
	if(MyResultVector == NULL)
		Status = 0;

	//Distributing the Matrix among to all the nodes
	MPI_Scatter(MatrixA, ScatterSize * ColsNo, MPI_FLOAT, MyMatrixA, ScatterSize * ColsNo, MPI_FLOAT, Root, MPI_COMM_WORLD);

	DeviceStatus = CheckDevice(MyRank);

        if(DeviceStatus == 0)
        {
                printf("Processor with rank %d doing partial product of two vectors on CPU \n",MyRank);
		for(Index = 0 ; Index < ScatterSize ; Index++) 
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
		CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMyMatrixA, ScatterSize * ColsNo * sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMyResultVector, ScatterSize * sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceVectorB, VectorSize * sizeof(float) ) );

		//Copying the data from host to device
		cudaMemcpy( (void *)DeviceMyMatrixA, (void *)MyMatrixA, ScatterSize * ColsNo * sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy( (void *)DeviceVectorB, (void *)VectorB, VectorSize * sizeof(float), cudaMemcpyHostToDevice );

		//Calling the kernel which performs Matrix Vector Product
		MatrixVectorMultiplication<<<DimGrid, DimBlock>>>(DeviceMyMatrixA, DeviceVectorB, DeviceMyResultVector, ColsNo, ScatterSize, BLOCKSIZE);	
	
		//Copying the value of patial result vector from device to host
		cudaMemcpy( (void *)MyResultVector, (void *)DeviceMyResultVector, ScatterSize * sizeof(float), cudaMemcpyDeviceToHost );
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	//Root processor gathering from all nodes to get the final result vector
	MPI_Gather( MyResultVector, ScatterSize, MPI_FLOAT, ResultVector, ScatterSize, MPI_FLOAT, Root, MPI_COMM_WORLD);

	//Root processor printing the resultant vector
	if(MyRank == Root)
	{
		printf("The resultant vector with size %d  is \n",RowsNo);
		for(Index = 0; Index < RowsNo; Index++)
			printf(" %f \n", ResultVector[Index]);

		//freeing the Vectors allocated by the root node
		free(MatrixA);
		free(ResultVector);
	}

	//Freeing the host memory	
	free(MyMatrixA);
	free(VectorB);
	free(MyResultVector);
	
	//Freeing the device memory
	CUDA_SAFE_CALL( cudaFree( DeviceMyMatrixA ) );
	CUDA_SAFE_CALL( cudaFree( DeviceVectorB ) );
	CUDA_SAFE_CALL( cudaFree( DeviceMyResultVector ) );

	MPI_Finalize();
	return(0);
}//End of Main function
//---------------------------------------------------------------------------------------------------------------------------------------

int IntializingMatrixVectors(float **MatrixA, float **VectorB, float **ResultVector, int RowsNo, int ColsNo, int VectorSize)
{
	float *TempMatrixA, *TempVectorB, *TempResultVector;
	int Status, Index;

	//Allocating memory on the host
	TempMatrixA = (float *)malloc(RowsNo * ColsNo * sizeof(float));
	if(TempMatrixA == NULL)
		Status = 0;
	TempVectorB = (float *)malloc(VectorSize * sizeof(float));
	if(TempVectorB == NULL)
		Status = 0;
	TempResultVector = (float *)malloc(RowsNo * sizeof(float));
	if(TempResultVector == NULL)
		Status = 0;

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
	
	return(Status);
}//End of the function
//-------------------------------------------------------------------------------------------------------------------------------------

int CheckDevice(int MyRank)
{
        int DeviceCount, Device;
        struct cudaDeviceProp Properties;

        cudaGetDeviceCount(&DeviceCount);
        if(DeviceCount >= 1)
        {
                cudaGetDevice(&Device);
                cudaGetDeviceProperties(&Properties, Device);
                printf("Processor with  rank %d has the Device by name %s and computation is done on this device \n",MyRank, Properties.name);
        }

        return(DeviceCount);

}//End of function
//--------------------------------------------------------------------------------------------------------------

