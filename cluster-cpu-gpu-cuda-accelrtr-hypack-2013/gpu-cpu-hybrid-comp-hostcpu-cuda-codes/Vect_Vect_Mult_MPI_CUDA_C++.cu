/****************************************************************************

                C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Objective   : program to solve a Vector Vector multiplication using block
                striped partioning on hybrid computing using MPI C++ and CUDA

  Input       : Process 0 initialize the Vector.

  Output      : Process 0 prints the resultant value.

  Necessary      Size of the  Each Vector should be greater than number of
  Conditons   :  processes and perfectly divisible by number of processes.

  Created     : August-2013

  E-mail      : hpcfte@cdac.in

****************************************************************************/

#include<iostream.h>
#include<stdlib.h>
#include<unistd.h>
#include<cuda.h>
#include"mpi.h"
//--------------------------------------------------------------------------------------------------------------

#define BLOCKSIZE 16

//---------------------------------------------------------------------------------------------------------------

int  IntializingVectors(float **, float **, int  );
int GetDeviceStatus(int );

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

//---------------------------------------------------------------------------------------------------------------------------
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
	int MyRank, NumberOfProcessors;
	float *VectorA, *VectorB, *MyVectorA, *MyVectorB, *NodeSum;
	float *DeviceMyVectorA, *DeviceMyVectorB, *DeviceTempVector, *DeviceNodeSum;
	int  VectorSize, ScatterSize, DeviceStatus, Root = 0, Status = 1, Index;
	float FinalResult;

	//MPI Initialization and getting the number of nodes and node id of this node
	MPI::Init(argc, argv);
	NumberOfProcessors = MPI::COMM_WORLD.Get_size();
	MyRank = MPI::COMM_WORLD.Get_rank();

	//Checking if valid number of inputs have been passed
	if(argc != 2)
        {
                if(MyRank == Root)
                        cout<<"Usage:<mpirun><-n><Number of Processors><./Program Name><Vector Size>"<<endl;
                MPI::Finalize();
                exit(-1);
        }

	//Getting Vector Size from program arguements
	VectorSize = atoi(argv[1]);

	if(VectorSize < NumberOfProcessors)
	{
		if(MyRank == Root)
			cout<<"Vector size should be more than number of processors ";
		MPI::Finalize();
		exit(-1);
	}

	//Checking if Data can be distributed evenly to all processors
	if(VectorSize % NumberOfProcessors != 0)
	{
		if(MyRank == Root)
			cout<<"Vector can not be distributed evenly among the nodes";
		MPI::Finalize();
		exit(-1);
	}

	 //Root Processor intializes the VectorA and VectorB
        if(MyRank == Root)
                Status = IntializingVectors(&VectorA, &VectorB, VectorSize);

	 MPI::COMM_WORLD.Bcast(&Status, 1, MPI::INT, Root);

        //Checking if status returned by the function IntializingVectors is zero
        if(Status == 0)
        {
                if(MyRank == Root)
                        printf("I am processor %d and the memory is not available for the varilable on the host \n",MyRank);
                MPI_Finalize();
                exit(-1);
        }

	//Computing the Scatter Size
	ScatterSize = VectorSize / NumberOfProcessors;
	
	//Memory allocation by all nodes
	MyVectorA = new float[ScatterSize];
	MyVectorB = new float[ScatterSize];
	NodeSum = new float[1];

	(*NodeSum) = 0.0f;

	//Root processor distributing the vectors A & B to all the processors
	MPI::COMM_WORLD.Scatter(VectorA, ScatterSize, MPI::FLOAT, MyVectorA, ScatterSize, MPI::FLOAT, Root);
	MPI::COMM_WORLD.Scatter(VectorB, ScatterSize, MPI::FLOAT, MyVectorB, ScatterSize, MPI::FLOAT, Root);

	DeviceStatus = GetDeviceStatus(MyRank);
	
	if(DeviceStatus == 0)
	{
		cout<<"Processor with rank" <<MyRank<< "doing partial product of two vectors on CPU" << endl;
		for(Index = 0; Index < ScatterSize; Index++)      
        		   (*NodeSum) = (*NodeSum) + (MyVectorA[Index] * MyVectorB[Index]);
	}
	else
	{
		//Allocating memory on the Device memory
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

	MPI::COMM_WORLD.Barrier();

        //Adding the NodeSum value from all the nodes to get the final product value
        MPI::COMM_WORLD.Reduce(NodeSum, &FinalResult, 1, MPI::FLOAT, MPI::SUM, Root);

	//Root node printing the product of given two vectors
        if(MyRank == Root)
        {
                cout<<"The product of the given two vectors is " << FinalResult << endl;
                //Freeing the Vectors allocated by root node
                delete(VectorA);
                delete(VectorB);
        }

	//Freeing the Host Memory
        delete(MyVectorA);
        delete(MyVectorB);
        delete(NodeSum);

        //Freeing  the Device Memory
        CUDA_SAFE_CALL( cudaFree(DeviceMyVectorA) );
        CUDA_SAFE_CALL( cudaFree(DeviceMyVectorB) );
        CUDA_SAFE_CALL( cudaFree(DeviceTempVector) );
        CUDA_SAFE_CALL( cudaFree(DeviceNodeSum) );

	//MPI finalization
	MPI::Finalize();

	return(0);
}//End of main function
//-------------------------------------------------------------------------------------------------------

int IntializingVectors(float **VectorA, float **VectorB, int  VectorSize)
{
	float *TempVectorA, *TempVectorB;
	int Index, Status = 1;

	//Allocating memory on the host by root processor
	TempVectorA = new float [VectorSize];
	if(TempVectorA == NULL)
		Status = 0;
	TempVectorB = new float [VectorSize];
	if(TempVectorB == NULL)
		Status = 0;

	//Assigning each element of the vector to one more than its index
	for(Index = 0; Index < VectorSize; Index++)
        {
                TempVectorA[Index] = Index + 1;
                TempVectorB[Index] = Index + 1;
        }

	//assigning address of temporary variables to real variables
        *VectorA = TempVectorA;
        *VectorB = TempVectorB;
	
	return(Status);
}//End of function.
//------------------------------------------------------------------------------------------------------

int GetDeviceStatus(int MyRank)
{
	int DeviceCount, Device;
	struct cudaDeviceProp Properties;
	
	cudaGetDeviceCount(&DeviceCount);
	if(DeviceCount >= 1)
	{
		cudaGetDevice(&Device);
		cudaGetDeviceProperties(&Properties, Device);
		cout<<" Processor with  rank "<< MyRank <<" has the Device by name  "<< Properties.name << " and computation is done on this device "<<endl;
	}

	return(DeviceCount);

}//End of function
//--------------------------------------------------------------------------------------------------------------

