/*********************************************************************

	C-DAC Tech Workshop : hyPACK-2013
               October 15-18, 2013

Objective   : program to solve a  Poisson Equation in a GPU
   
Input       : No of Points in X-Direction, No of Points in Y-Direction & 
              and maximum number of iterations
   
Output      : Solution Vector.

Created     : August-2013

E-mail      : hpcfte@cdac.in     

*************************************************************************/
#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
#include<sys/time.h>
#include"cublas.h"

//-------------------------------------------------------------------------

#define BLOCKSIZE 16
#define TOLERANCE 1.0E-06
#define TOPBOUNDARYVALUE  4.1f
#define BOTTOMBOUNDARYVALUE  3.1f
#define LEFTBOUNDARYVALUE  1.1f
#define RIGHTBOUNDARYVALUE  2.1f

//-----------------------------------------------------------------------
void IntializeAndSetBoundaryConditions(float **, float **, int , int , int );
void SetBoundaryCondition(int , int , float , int , float *, float *);
void IntializeUInteriorIndex(int **, int , int , int );
void IntializeUDifference(float **, int );

//------------------------------------------------------------------------

//Pragma routine to report the detail of cuda error
#define CUDA_SAFE_CALL(call)                                                          \
	do{                                                                           \
          	cudaError_t err = call;                                               \
                if(err != cudaSuccess)                                                \
                {                                                                     \
                	fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                        __FILE__, __LINE__, cudaGetErrorString( err) );               \
                        exit(1);                                                      \
                }                                                                     \
          } while (0)                                                                 \
//-----------------------------------------------------------------------------------  

//Kernel that performs the Jacobi Iteration
__global__ void JacobiIteration(float *DeviceUOld, float *DeviceUNew, int *DeviceUInteriorIndex, int NoPointsX, int Size, int ThreadDim)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int ThreadIndex = (ThreadDim * tidx) + tidy;
	int MaxNumThread = ThreadDim * ThreadDim;
	int CurrentColumnIndex;
	int pass = 0;
	int Center, Left, Right, Bottom, Top;

	while( (CurrentColumnIndex = (ThreadIndex + MaxNumThread * pass)) < Size )
	{
		Center = DeviceUInteriorIndex[CurrentColumnIndex];
		Left   = Center - 1;
		Right  = Center + 1;
		Top    = Center - NoPointsX;
		Bottom = Center + NoPointsX;

		//Updating the UNew values
		DeviceUNew[Center] = 0.25 * (DeviceUOld[Left] + DeviceUOld[Right] + DeviceUOld[Top] + DeviceUOld[Bottom]);
	
		pass++;
	}
	__syncthreads();
}//End of Jacobi Iteration Device function
//------------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
	//Checking if valid number of Arguements have been passed
	if(argc != 4)
	{
		printf("Valid number of inputs are not given \n");
		printf("Usage:<./Program Name><Number of X points><Number of Y points><Maximum Number of Iterations> \n");
		exit(-1);
	}

	//Host Variables Declaration
	float *UOld, *UNew, *UDifference;
	int *UInteriorIndex;
	float MaxError = 0.0f;
	struct timeval TV;
	double StartTime,EndTime,ActualTime;
	int NoPointsX, NoPointsY, MaxIterations, NoPointsInterior, Index, PresentIteration,NoPointsTotal;

	//Device Variables Declaration
	float *DeviceUOld, *DeviceUNew, *DeviceUDifference;
	int *DeviceUInteriorIndex;

	//Obtaining the Values of NoPointsX, NoPointsY and MaxIterations from the arguements passed by the User
	NoPointsX = atoi( argv[1] );
	NoPointsY = atoi( argv[2] );
	MaxIterations = atoi( argv[3] );

	//Calculating the Total Points and Interior Points
	NoPointsTotal = NoPointsX * NoPointsY;
	NoPointsInterior = (NoPointsTotal) - (((2 * NoPointsX) + (2 * NoPointsY)) - 4);

	//Intializing the UOld and seting the Boundary conditions
	IntializeAndSetBoundaryConditions( &UOld, &UNew, NoPointsX, NoPointsY, NoPointsTotal );

	//Intializing the UDifference
	IntializeUDifference( &UDifference,NoPointsTotal );

	//Filling the UInteriorIndex with Index Values of Interior Points
	IntializeUInteriorIndex( &UInteriorIndex, NoPointsX, NoPointsY,NoPointsInterior ); 
	

	//Allocating Memory on Device
	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceUOld, NoPointsTotal * sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceUNew, NoPointsTotal * sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceUInteriorIndex, NoPointsInterior * sizeof(int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceUDifference, NoPointsTotal *  sizeof(float) ) );

	//Copying Data from Host to Device
	CUDA_SAFE_CALL( cudaMemcpy((void *)DeviceUOld, (void *)UOld, NoPointsTotal * sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy((void *)DeviceUNew, (void *)UNew, NoPointsTotal * sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy((void *)DeviceUInteriorIndex, (void *)UInteriorIndex, NoPointsInterior * sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy((void *)DeviceUDifference, (void *)UDifference, NoPointsTotal * sizeof(float), cudaMemcpyHostToDevice) );

	//Defining Thread Grid and the Thread Block
	dim3 DimGrid( 1,1 );
	dim3 DimBlock( BLOCKSIZE,BLOCKSIZE );

	PresentIteration = 0;

	//start timing computation
        gettimeofday(&TV, NULL);
         StartTime = TV.tv_sec+( TV.tv_usec/1000000.0 );

	while(1)
	{
		//Incrementing the Iteration Number
		PresentIteration++;

		//Invoking the Kernel
		JacobiIteration<<<DimGrid, DimBlock>>>( DeviceUOld, DeviceUNew, DeviceUInteriorIndex, NoPointsX, NoPointsInterior, BLOCKSIZE );
	
		//Copying the UNew to UDifference
		cublasScopy( NoPointsTotal, DeviceUNew, 1, DeviceUDifference, 1 );

		//Finding the Difference between UNew and UOld
		cublasSaxpy( NoPointsTotal, -1.0f, DeviceUOld, 1, DeviceUDifference, 1 );

		//Assigning UNew to UOld
		cublasScopy( NoPointsTotal, DeviceUNew, 1, DeviceUOld, 1 );	

		//Copying Udifference from Device to Host
		CUDA_SAFE_CALL( cudaMemcpy((void *)UDifference, (void *)DeviceUDifference, NoPointsTotal * sizeof(float), cudaMemcpyDeviceToHost ) );

		//Finding the Maximum among the UDifference values	
		Index = cublasIsamax(NoPointsTotal, DeviceUDifference, 1);
 		MaxError = UDifference[ (Index - 1) ];

		//Checking for the convergence
		if((MaxError < TOLERANCE) ||  (PresentIteration == MaxIterations))
			break;
	}

	//stop timing computation
        gettimeofday(&TV,NULL);
        EndTime = TV.tv_sec+(TV.tv_usec/1000000.0);
 
         //calculate difference between start and stop times
         ActualTime = EndTime - StartTime;

	//Copying UNew from Device to Host
	CUDA_SAFE_CALL(cudaMemcpy((void *)UNew, (void *)DeviceUNew, NoPointsTotal * sizeof(float), cudaMemcpyDeviceToHost));
	
	//Printing the solution
	for(Index = 0; Index < NoPointsTotal; Index++)
		printf(" \t %d \t %f \n", Index, UNew[Index]);

	printf("Output Vector given above calculated in %d Iterations and in %lf secs.\n",PresentIteration,ActualTime);

	//Freeing the Allocated Memory on Device
	CUDA_SAFE_CALL( cudaFree( DeviceUOld ) );
	CUDA_SAFE_CALL( cudaFree( DeviceUNew ) );
	CUDA_SAFE_CALL( cudaFree( DeviceUInteriorIndex ) );
	CUDA_SAFE_CALL( cudaFree( DeviceUDifference ) );

	//Freeing the Allocated Memory on Host
	free( UOld );
	free( UNew );
	free( UInteriorIndex );
	free( UDifference );

	return(0);
}//End of Main
//------------------------------------------------------------------------------------------------------------

void IntializeAndSetBoundaryConditions( float **UOld, float **UNew, int NoPointsX, int NoPointsY, int NoPointsTotal )
{
	float *TempUOld,*TempUNew;
	int Index;
	//Allocating memory for UOld and UNew
	TempUOld = (float *)malloc( NoPointsTotal * sizeof(float) );
	if(TempUOld == NULL)
	{
		printf("Can't allocate the memory for the variable TempUOld \n");
		exit(-1);
	}
	TempUNew = (float *)malloc( NoPointsTotal * sizeof(float) );
	if(TempUNew == NULL)
	{
		printf("Can't allocate the memory for the variable TempUNew \n");
		exit(-1);
	}
	
	//Intialize UOld to zeros
	for(Index = 0; Index < (NoPointsTotal); Index++)
		TempUOld[Index] = 0.0;

	//Setting the Boundary Conditions

	//Case:Left
	for(Index = 0; Index < NoPointsY; Index++)
		SetBoundaryCondition(0, Index, LEFTBOUNDARYVALUE, NoPointsX, TempUOld, TempUNew);

	//Case:Right
	for(Index = 0; Index < NoPointsY; Index++)
		SetBoundaryCondition((NoPointsX - 1), Index, RIGHTBOUNDARYVALUE, NoPointsX, TempUOld, TempUNew);

	//Case:Bottom
	for(Index = 0; Index < NoPointsX; Index++)
		SetBoundaryCondition(Index, 0, BOTTOMBOUNDARYVALUE, NoPointsX, TempUOld, TempUNew);

	//Case:Top
	for(Index = 0; Index < NoPointsX; Index++)
		SetBoundaryCondition(Index, (NoPointsY - 1), TOPBOUNDARYVALUE, NoPointsX, TempUOld, TempUNew);

	//Assigning Temporary Varibles Locations to Original Variables
	*UOld = TempUOld;
	*UNew = TempUNew;
}
//---------------------------------------------------------------------------------------------------

void SetBoundaryCondition(int i, int j, float Value, int NoPointsX, float *UOld, float *UNew)
{
	int Index;
	Index = (j * NoPointsX) + i;
	UOld[Index] = Value;
	UNew[Index] = Value;
}
//-------------------------------------------------------------------------------------------------------

void IntializeUInteriorIndex(int **UInteriorIndex, int NoPointsX, int NoPointsY,int NoPointsInterior)
{
	int i, j, Index, IndexValue;
	int *TempUInteriorIndex;
	Index = 0;

	//Allocating memory for UInteriorIndex
	TempUInteriorIndex = (int *)malloc( NoPointsInterior * sizeof(int) );
	if( TempUInteriorIndex == NULL )
	{
		printf("Can't allocate memory for the variable TempUInteriorIndex \n");
		exit(-1);
	}

	//Assigning the index of the Interior points of UOld and UNew
	for(j = 1; j < (NoPointsY - 1); ++j)
	{
		for(i = 1; i < (NoPointsX - 1); i++)
		{
			IndexValue = (j * NoPointsX) + i;
			TempUInteriorIndex[Index] = IndexValue;
			Index++;
		}
	}
	
	*UInteriorIndex = TempUInteriorIndex;
}
//------------------------------------------------------------------------------------------------------

void IntializeUDifference(float **UDifference, int NoPointsTotal)
{
	float *TempUDifference;
	int RowNumber;
	
	//Allocating Memory for UDifference
	TempUDifference = (float *)malloc( NoPointsTotal * sizeof(float) );
	if( TempUDifference == NULL )
	{
		printf("Can't allocate the memory for the variable TempUDifference \n");
		exit(-1);
	}

	//Intializing the UDifference to zero's
	for(RowNumber = 0; RowNumber < NoPointsTotal; RowNumber++)
		TempUDifference[RowNumber] = 0.0f;

	*UDifference = TempUDifference;
}
//--------------------------------------------------------------------------------------------------------
