/****************************************************************************

                C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Objective   : program to solve a Vector Vector multiplication using block
                striped partioning on hybrid computing using MPI and CUDA

  Input       : Process 0 initialize the Vector.

  Output      : Process 0 prints the resultant value.

  Necessary      Size of the  Each Vector should be greater than number of
  Conditons   :  processes and perfectly divisible by number of processes.

  Created     : August-2013

  E-mail      : hpcfte@cdac.in

****************************************************************************/

#include<stdio.h>
#include<cuda.h>
#include"mpi.h"
//---------------------------------------------------------------------------------------------------------------

#define BLOCKSIZE 16

//---------------------------------------------------------------------------------------------------------------

int  IntializingVectors(float **, float **, int  );
int CheckDevice(int );

//----------------------------------------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------------------------------------

//Kernel that performs Vector Vector Dot Product
__global__ void VectorVectorDotProduct(float *Vector1,float *Vector2,float *TempVector,float *Solution,int VectorSize,int ThreadDim)
{
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;
        int ThreadIndex = (ThreadDim * tidx) + tidy;
        int MaxNumThread = ThreadDim * ThreadDim;
        int ThreadColumnIndex,RowNum,pass = 0;
        *Solution = 0.0;
        while( (ThreadColumnIndex = (ThreadIndex + MaxNumThread * pass))  < VectorSize )
        {
                TempVector[ThreadColumnIndex] = (Vector1[ThreadColumnIndex] * Vector2[ThreadColumnIndex]) ;
                pass++;
        }
        __syncthreads();
        if(ThreadIndex == 0)
        {
                for(RowNum=0;RowNum<VectorSize;RowNum++)
                        *Solution += TempVector[RowNum];
        }
}//End of Vector Vector Dot product Device Function
//-------------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
	int MyRank, NumberOfProcessors, VectorSize, ScatterSize;
	float *VectorA, *VectorB;
	float *MyVectorA, *MyVectorB;
	float *DeviceMyVectorA, *DeviceMyVectorB, *DeviceTempVector;
	float *NodeSum, *DeviceNodeSum;
	float Result;
	int DeviceStatus, Index, Root = 0, Status = 1;

	//Intilaizing the MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
	MPI_Comm_size(MPI_COMM_WORLD, &NumberOfProcessors);

	//Checking if valid number of arguements have been passed
	if(argc != 2)
	{
		if(MyRank == Root)
			printf("Usage:<mpirun><-n><Number of Processors><./Program Name><Vector Size> \n");	
		MPI_Finalize();
		exit(0);
	}	

	//Getting Vector Size program arguements
	VectorSize = atoi(argv[1]);
	
	//Checking if Vector size is less than the Total number of processors
	if(VectorSize < NumberOfProcessors)
	{
		MPI_Finalize();
		if(MyRank == Root)
		{
			printf("Vector Size should be more than Number of processors \n");
			exit(-1);
		}
		exit(-1);
	}
	
	//Checking if data can be distributed evenly to all nodes 
	if(VectorSize % NumberOfProcessors != 0)
	{
		MPI_Finalize();
		if(MyRank == Root)
			printf("Vectos can not be distributed evenly among all processors \n");
		exit(-1);
	}

	//Root node intializes the VectorA and VectorB
        if(MyRank == Root)
                Status = IntializingVectors(&VectorA, &VectorB, VectorSize);

        MPI_Bcast(&Status, 1, MPI_INT, Root, MPI_COMM_WORLD);

        //Checking if status returned by the function IntializingVectors is zero
        if(Status == 0)
        {
                if(MyRank == Root)
                        printf("I am processor %d and the memory is not available for the varilable on the host \n",MyRank);
                MPI_Finalize();
                exit(-1);
        }
		
	//Calculating the Scatter size
	ScatterSize = VectorSize / NumberOfProcessors;
	
	//Allocating memory on the host by all the nodes
	MyVectorA = (float *)malloc(ScatterSize * sizeof(float));
	MyVectorB = (float *)malloc(ScatterSize * sizeof(float));
	NodeSum = (float *)malloc(sizeof(float));

	//Distributing the VectorA and VectorB to all the nodes
	MPI_Scatter(VectorA, ScatterSize, MPI_FLOAT, MyVectorA, ScatterSize, MPI_FLOAT, Root, MPI_COMM_WORLD);
	MPI_Scatter(VectorB, ScatterSize, MPI_FLOAT, MyVectorB, ScatterSize, MPI_FLOAT, Root, MPI_COMM_WORLD);

	DeviceStatus = CheckDevice(MyRank);

	if(DeviceStatus == 0)
        {
                printf("Processor with rank %d doing partial product of two vectors on CPU \n",MyRank);
                for(Index = 0; Index < ScatterSize; Index++)
                           (*NodeSum) = (*NodeSum) + (MyVectorA[Index] * MyVectorB[Index]);
        }
	else
	{
		//allocating memory on the Device memory
		CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMyVectorA, ScatterSize * sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMyVectorB, ScatterSize * sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceTempVector, ScatterSize * sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceNodeSum, sizeof(float) ) );

		//Copying data from Host to device memory
		CUDA_SAFE_CALL( cudaMemcpy( (void *)DeviceMyVectorA, (void *)MyVectorA, ScatterSize * sizeof(float), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( (void *)DeviceMyVectorB, (void *)MyVectorB, ScatterSize * sizeof(float), cudaMemcpyHostToDevice) );

		//Defining Thread Grid and Thread Block
		dim3 DimGrid(1,1);
		dim3 DimBlock(BLOCKSIZE, BLOCKSIZE);
	
		//Calling the kernel which performs Vector Vector Product
		VectorVectorDotProduct<<<DimGrid, DimBlock>>>(DeviceMyVectorA, DeviceMyVectorB, DeviceTempVector,DeviceNodeSum, ScatterSize, BLOCKSIZE);
	
		//Copying the value of the node sum from the Device to the Host
		CUDA_SAFE_CALL( cudaMemcpy((void *)NodeSum, (void *)DeviceNodeSum, sizeof(float), cudaMemcpyDeviceToHost) ); 	
	}
	
	MPI_Barrier(MPI_COMM_WORLD);

	//Adding the NodeSum value from all the nodes to get the final product value 
	MPI_Reduce(NodeSum, &Result, 1, MPI_FLOAT, MPI_SUM, Root, MPI_COMM_WORLD);

	//Root node printing the product of given two vectors 
	if(MyRank == Root)
	{
		printf("The product of the given two vectors is %f \n", Result);
		//Freeing the Vectors allocated by root node
		free(VectorA);
		free(VectorB);
	}

	//Freeing the Host Memory
	free(MyVectorA);
	free(MyVectorB);
	free(NodeSum);

	//Freeing  the Device Memory
	CUDA_SAFE_CALL( cudaFree(DeviceMyVectorA) );
	CUDA_SAFE_CALL( cudaFree(DeviceMyVectorB) );
	CUDA_SAFE_CALL( cudaFree(DeviceTempVector) );
	CUDA_SAFE_CALL( cudaFree(DeviceNodeSum) );	
		
	MPI_Finalize();

	return(0);

}//End of Main function
//------------------------------------------------------------------------------------------------------------------------------------

int  IntializingVectors(float **VectorA, float **VectorB, int  VectorSize)
{
	float *TempVectorA, *TempVectorB;
	int Index, Status = 1;

	TempVectorA = (float *)malloc(VectorSize * sizeof(float));
	if(TempVectorA == NULL)
		Status = 0;
	
	TempVectorB = (float *)malloc(VectorSize * sizeof(float));
	if(TempVectorB == NULL)	
		Status = 0;
	
	for(Index = 0; Index < VectorSize; Index++)
	{
		TempVectorA[Index] = Index + 1;
		TempVectorB[Index] = Index + 1;
	}
	
	*VectorA = TempVectorA;
	*VectorB = TempVectorB;
	
	return(Status);
}//End of function
//----------------------------------------------------------------------------------------------------------------------------------

int CheckDevice(int MyRank)
{
        int DeviceCount, Device;
        struct cudaDeviceProp Properties;

        cudaGetDeviceCount(&DeviceCount);
        if(DeviceCount >= 1)
        {
                cudaGetDevice(&Device);
                cudaGetDeviceProperties(&Properties, Device);
                printf("Processor with  rank %d has the Device by name %s and compuatation is doen on this device\n",MyRank, Properties.name);
        }

        return(DeviceCount);

}//End of function
//--------------------------------------------------------------------------------------------------------------
