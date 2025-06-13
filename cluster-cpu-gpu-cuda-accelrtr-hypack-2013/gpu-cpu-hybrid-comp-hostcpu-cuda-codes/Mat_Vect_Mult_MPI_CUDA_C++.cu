/****************************************************************************

                C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Objective   : program to solve a Matrix vector multiplication using block
                striped partioning on hybrid computing using MPI C++ and CUDA

  Input       : Matrix Rows, Matrix Columns, Vector Size and Process 0
                intializes the Matrix and the vector.

  Output      : Process 0 prints the resultant vector.

  Necessary      Number of rows of matrix should be greater than number of
  Conditons   :  processes and perfectly divisible by number of processes.
                 Matrix columns  should be equal to vector size.

  Created     : August-2013

  E-mail      : hpcfte@cdac.in

****************************************************************************/

#include<iostream.h>
#include<stdlib.h>
#include<unistd.h>
#include<cuda.h>
#include<mpi.h>
//-------------------------------------------------------------------------------------------------------------------------------------------

#define BLOCKSIZE 16

//-------------------------------------------------------------------------------------------------------------------------------------------

int IntializingMatrixVectors(float **, float **, float **, int , int , int );
int CheckDevice(int );

//-------------------------------------------------------------------------------------------------------------------------------------------
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

	//MPI Initialization and getting the number of nodes and node id of this node
        MPI::Init(argc, argv);
        NumberOfProcessors = MPI::COMM_WORLD.Get_size();
        MyRank = MPI::COMM_WORLD.Get_rank();

	//Checking if valid number of arguements have been passed
        if(argc != 4)
        {
                if(MyRank == Root)
                        cout<<"Usage:<mpirun><-n><Number of processors><./Program Name><Number of Rows of Matrix><Number of Columns of Matrix><VectorSize>"<<endl;
                MPI::Finalize();
                exit(-1);
        }

	//Getting  RowsNo, ColsNo, VectorSize from the program arguements 
        RowsNo = atoi( argv[1] );
        ColsNo = atoi( argv[2] );
        VectorSize = atoi( argv[3] );

	//Checking if columns is equal to vector size
        if(ColsNo != VectorSize)
        {
                if(MyRank == Root)
                        cout<<"Entered wrong input, Number of columns of matrix should be equal to size of the vector "<<endl;
                MPI::Finalize();
                exit(-1);
        }
	
	 if(RowsNo < NumberOfProcessors)
        {
                if(MyRank == Root)
                        cout<<"Given number of Rows of the matrix should be more than number of processors "<<endl;
                MPI::Finalize();
                exit(-1);
        }

        //Checking if Matrix can be distributed evenly to all the nodes
        if(RowsNo % NumberOfProcessors != 0)
        {
                if(MyRank == Root)
                        cout<<"The Rows of the matrix can not be distributed evenly among processors "<<endl;
                MPI::Finalize();
                exit(-1);
        }
	
	//Root node intializes the Matrix, Vector and Result Vector
        if(MyRank == Root)
                Status = IntializingMatrixVectors(&MatrixA, &VectorB, &ResultVector, RowsNo, ColsNo, VectorSize);

        MPI::COMM_WORLD.Barrier();

        MPI::COMM_WORLD.Bcast(&Status, 1, MPI::INT, Root);

        //Checking if Status returned by the IntializingMatrixVectors is zero
        if(Status == 0)
        {
                if(MyRank == Root)
                        cout<<"Memory is not available to allocate for the Variables "<<endl;
                MPI::Finalize();
                exit(-1);
        }
	
	//Allocating memory for the VectorB by all nodes expect root node
        if(MyRank != Root)
                VectorB = new float[VectorSize];

	//Broadcasting the Vector to all the nodes from root node
        MPI::COMM_WORLD.Bcast(VectorB, VectorSize, MPI::FLOAT, Root);

	//Calculating the Scatter size of the Matrix
        ScatterSize = RowsNo / NumberOfProcessors;

        //Allocating the memory on the host for the MyMatrixA and MyResultVector by all nodes
        MyMatrixA = new float[ScatterSize * ColsNo];
        MyResultVector = new float[ScatterSize];

	//Distributing the Matrix  to all the nodes by the root node
        MPI::COMM_WORLD.Scatter(MatrixA, ScatterSize * ColsNo, MPI::FLOAT, MyMatrixA, ScatterSize * ColsNo, MPI::FLOAT, Root);

	//Getting the status of the device
        DeviceStatus = CheckDevice(MyRank);

	//Checking if Device Status is zero
	if(DeviceStatus == 0)
        {
                cout<<"Processor with rank  "<<MyRank<< " doing partial product of two vectors on CPU "<<endl;
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

	MPI::COMM_WORLD.Barrier();
	
	//Root processor gathering from all nodes to get the final result vector
        MPI::COMM_WORLD.Gather( MyResultVector, ScatterSize, MPI::FLOAT, ResultVector, ScatterSize, MPI::FLOAT, Root);

	//Root processor printing the resultant vector
        if(MyRank == Root)
        {
                cout<<"The resultant vector with size  " <<RowsNo <<" is "<<endl;
                for(Index = 0; Index < RowsNo; Index++)
                        cout<< ResultVector[Index] <<endl;

                //freeing the Vectors allocated by the root node
                delete(MatrixA);
                delete(ResultVector);
        }

	 //Freeing the host memory
        delete(MyMatrixA);
        delete(VectorB);
        delete(MyResultVector);

        //Freeing the device memory
        CUDA_SAFE_CALL( cudaFree( DeviceMyMatrixA ) );
        CUDA_SAFE_CALL( cudaFree( DeviceVectorB ) );
        CUDA_SAFE_CALL( cudaFree( DeviceMyResultVector ) );

	//MPI Finalization
	MPI::Finalize();

	return(0);

}//End of main function
//-----------------------------------------------------------------------------------------------------------------------------------------------
int IntializingMatrixVectors(float **MatrixA, float **VectorB, float **ResultVector, int RowsNo, int ColsNo, int VectorSize)
{
        float *TempMatrixA, *TempVectorB, *TempResultVector;
        int Status, Index;

        //Allocating memory on the host
        TempMatrixA = new float[RowsNo * ColsNo];
        if(TempMatrixA == NULL)
                Status = 0;
        TempVectorB = new float[VectorSize] ;
        if(TempVectorB == NULL)
                Status = 0;
        TempResultVector = new float[RowsNo];
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

	//Getting count of the devices available
        cudaGetDeviceCount(&DeviceCount);

        if(DeviceCount >= 1)
        {
		//Getting device id
                cudaGetDevice(&Device);

		//Getting properties of the device
                cudaGetDeviceProperties(&Properties, Device);

                cout<<"Processor with rank "<<MyRank<<"  has the Device by name  "<< Properties.name <<" and computation is done on this device "<<endl;
        }

        return(DeviceCount);

}//End of function
//--------------------------------------------------------------------------------------------------------------

