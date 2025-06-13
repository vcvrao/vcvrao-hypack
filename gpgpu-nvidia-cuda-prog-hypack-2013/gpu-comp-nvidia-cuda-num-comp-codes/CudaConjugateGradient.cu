
/************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

    Objective : Program to solve a  linear system of equations (Ax = b) 
                using conjugate gradient method in a GPU

   Input      : Number of unknowns and max no. of iterations

   Output     : solution Vector 

   Created    : August-2013

   E-mail     : hpcfte@cdac.in     

****************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<cuda.h>
//--------------------------------------------------------------------------------
#define BLOCKSIZE 16
#define EPSILON 1.0E-20
//----------------------------------------------------------------------------------
void GetTheInputToTheProblem(float **,float **,float **,int );
void GetPreConditionMatrix(float ** ,int );
void SolvePreConditionMatrix(float *,float *,float *,int );
void GenerateCoefficientMatrix(float **,int);
void ConvertMatrixToOneDimension(float **,float *,int );
void GenerateRhsVector(float **,float *,int ); 
void IntializeDirectionVector(float ** ,int ,float * ,float * ,dim3 ,dim3 );
void  CalculateResidueVector(float ** ,float * ,float * ,float * ,int ,    float * ,float * ,dim3 ,dim3 );
void FreeTheDeviceMemory(float *,float *,float *,float *,float *,float *,float *,float *,float *,float *,float *);
void  FreeTheHostMemory(float *,float *,float *,float *,float *,float *,float *,float *,float *);
void *malloc_safe_call(int size);
//----------------------------------------------------------------------------------

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

//----------------------------------------------------------------------------------------

//Kernel that performs Matrix Vector Multiplication
__global__ void MatrixVectorMultiplication(float *Matrix,float *Vector,float *Solution,int MatrixSize,int ThreadDim)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
      	int ThreadIndex = (ThreadDim * tidx) + tidy;
      	int MaxNumThread = ThreadDim * ThreadDim;
      	int VectLen = MatrixSize;
      	int count,ThreadColumnIndex,pass = 0 ;
      	float TempResult = 0.0f;
        while( (ThreadColumnIndex = (ThreadIndex + MaxNumThread * pass))  <     MatrixSize )
      	{
        	TempResult = 0.0f;
          	for( count = 0; count < VectLen; count++)
               		TempResult +=  Matrix[ThreadColumnIndex*MatrixSize+count] * Vector[count];
          	Solution[ThreadColumnIndex] = TempResult;
          	pass++;
      	}
      	__syncthreads();
}//End of Matrix Vector Multiplication Device Function
//-----------------------------------------------------------------------------------------------------

//Kernel that performs Vector Matrix Multiplication
__global__ void VectorMatrixMultiplication(float *Matrix,float *Vector,float *Solution,int MatrixSize,int ThreadDim)
{
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
        int ThreadIndex = (ThreadDim * tidx) + tidy;
        int MaxNumThread = ThreadDim * ThreadDim;
        int VectLen = MatrixSize;
        int count,ThreadColumnIndex,pass = 0 ;
        float TempResult = 0.0f;
        while( (ThreadColumnIndex = (ThreadIndex + MaxNumThread * pass))  <     MatrixSize )
        {
        	TempResult = 0.0f;
                for( count = 0; count < VectLen; count++)
                	TempResult += Vector[count] * Matrix[MatrixSize * count + ThreadColumnIndex];
                 Solution[ThreadColumnIndex] = TempResult;
                 pass++;
         }
         __syncthreads();
}//End of Vector Matrix Mutiplication Device Function
//--------------------------------------------------------------------------------------------------

//Kernel that performs Vector Vector Subtraction
__global__ void VectorVectorSubtraction(float *Vector1,float *Vector2,float *Solution,int VectorSize,int ThreadDim)
{
	int tidx = threadIdx.x;
      	int tidy = threadIdx.y;
      	int ThreadIndex = (ThreadDim * tidx) + tidy;
      	int MaxNumThread = ThreadDim * ThreadDim;
      	int ThreadColumnIndex,pass = 0;
        while( (ThreadColumnIndex = (ThreadIndex + MaxNumThread * pass))  <     VectorSize )
      	{
        	Solution[ThreadColumnIndex] = (Vector1[ThreadColumnIndex] - Vector2[ThreadColumnIndex]) ;
          	pass++;
      	}
      	__syncthreads();
}//End of Vector Vector Subtraction Device Function
//---------------------------------------------------------------------------------------------------

//Kernel that performs Vector Vector Addition
__global__ void VectorVectorAddition(float *Vector1,float *Vector2,int VectorSize,int ThreadDim)
{
	int tidx = threadIdx.x;
        int tidy = threadIdx.y;
        int ThreadIndex = (ThreadDim * tidx) + tidy;
        int MaxNumThread = ThreadDim * ThreadDim;
        int ThreadColumnIndex,pass = 0;
        while( (ThreadColumnIndex = (ThreadIndex + MaxNumThread * pass))  <     VectorSize )
        {
                Vector1[ThreadColumnIndex] = (Vector1[ThreadColumnIndex] + Vector2[ThreadColumnIndex]) ;
                pass++;
        }
        __syncthreads();
}//End of Vector Vector Addition Device Function
//----------------------------------------------------------------------------------------------------------------------

//Kernel that subtracts each component of a vector from a given scalar
__global__ void ScalarVectorSubtraction(float *SolutionVector,float *Vector,float Scalar,int VectorSize,int ThreadDim)
{
	int tidx = threadIdx.x;
        int tidy = threadIdx.y;
        int ThreadIndex = (ThreadDim * tidx) + tidy;
        int MaxNumThread = ThreadDim * ThreadDim;
        int pass = 0;
        int ThreadColumnIndex;
        while( (ThreadColumnIndex = (ThreadIndex + MaxNumThread * pass))  <     VectorSize )
        	{
                	SolutionVector[ThreadColumnIndex] = (Scalar - Vector[ThreadColumnIndex]) ;
                 	pass++;
          	}
          	__syncthreads();
}//End of Scalar Vector Subtraction Device Function
//---------------------------------------------------------------------------------------------------------------------------------

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
//----------------------------------------------------------------------------------------------------------------------------------

//Kernel that performs Scalar Vector Multiplication
__global__ void ScalarVectorMultiplication(float *Vector,float Scalar,float *SolutionVector,int VectorSize,int ThreadDim)
{
	int tidx = threadIdx.x;
        int tidy = threadIdx.y;
        int ThreadIndex = (ThreadDim * tidx) + tidy;
        int MaxNumThread = ThreadDim * ThreadDim;
        int ThreadColumnIndex,pass = 0;
        while( (ThreadColumnIndex = (ThreadIndex + MaxNumThread * pass))  < VectorSize )
        {
   	     SolutionVector[ThreadColumnIndex] = Scalar * Vector[ThreadColumnIndex] ;
             pass++;
        }
        __syncthreads();
}//End of Scalar Vector Mutiplication Device Function
//------------------------------------------------------------------------------------------------------------------------

//Kernel that performs Scalar Matrix Multiplication
__global__ void ScalarMatrixMultiplication(float *Matrix,float Scalar,float *SolutionMatrix,int MatrixSize,int ThreadDim)
{
	int tidx = threadIdx.x;
        int tidy = threadIdx.y;
        int ThreadIndex = (ThreadDim * tidx) + tidy;
        int MaxNumThread = ThreadDim * ThreadDim;
        int pass = 0;
        int ThreadColumnIndex;
        while( (ThreadColumnIndex = (ThreadIndex + MaxNumThread * pass))  < MatrixSize*MatrixSize )
        {
             SolutionMatrix[ThreadColumnIndex] = Scalar * Matrix[ThreadColumnIndex] ;
             pass++;
        }
        __syncthreads();
}//End of Scalar Matrix Multiplication Device Function
//--------------------------------------------------------------------------------------------------------------------------

int main(int argc,char **argv)
{
	//Checking if  Valid number of Arguements have been passed
	if(argc != 3)
	{
		printf("Usage:<./a.out><Size of the Matrix><Maximum Number of Iterations> \n");
		exit(-1);
	}

	//Host Variables Declaration
	float *RhsVector,*SolutionVector,*CoefficientMatrix,*ResidueVector;
	float *PreConditionMatrix,*HVector,*DirectionVector;
	int    MaxIterations,Size,PresentIteration,RowNum;
	float *Delta0,*TempValue;
	float *Delta1;
	float Tau,Beta = 0.0;
	struct timeval tv;
	double cputime;
	
	//Device Variables Declaration
	float *DeviceRhsVector,*DeviceSolutionVector,*DeviceCoefficientMatrix,*DeviceResidueVector;
	float *DeviceHVector,*DeviceDirectionVector;
	float *DeviceDelta0,*DeviceTempValue,*DeviceTempMatrix,*DeviceTempBuffer;
	float *DeviceDelta1;

	//Obtaining the Size of the Matrix and Maximum number of Iterations from the given Arguements
	Size = atoi(argv[1]);
	MaxIterations = atoi(argv[2]);
	
	// Generating and Intializing the required Vectors and Matrix in the Host 
	GetTheInputToTheProblem(&RhsVector,&SolutionVector,&CoefficientMatrix,Size);
	
	//Allocating Memory on Device
	CUDA_SAFE_CALL(cudaMalloc( (void **)&DeviceRhsVector,Size * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc( (void **)&DeviceSolutionVector,Size * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc( (void **)&DeviceCoefficientMatrix,Size * Size * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc( (void **)&DeviceResidueVector,Size * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc( (void **)&DeviceTempBuffer,Size * sizeof(float)));
	
        //Copying Data from Host To Device
	CUDA_SAFE_CALL(cudaMemcpy((void*)DeviceRhsVector,(void*)RhsVector,Size*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy((void*)DeviceSolutionVector,(void*)SolutionVector,Size*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy((void*)DeviceCoefficientMatrix,(void*)CoefficientMatrix,Size*Size*sizeof(float),cudaMemcpyHostToDevice));
	
	//Defining Thread Grid and the Thread Block
	dim3 DimGrid(1,1);
	dim3 DimBlock(BLOCKSIZE,BLOCKSIZE);
	
	//Starting timing the computation
	gettimeofday(&tv, NULL);
   	double t1=tv.tv_sec+(tv.tv_usec/1000000.0);

	//Calculate the Residue vector RESIDUE=AX-B
	CalculateResidueVector(&ResidueVector,DeviceCoefficientMatrix,DeviceSolutionVector,DeviceTempBuffer,Size,DeviceRhsVector,DeviceResidueVector,DimGrid,DimBlock);
	
	//Getting Pre-Condional Matrix, Precondioanal Matrix is Identity Matrix	
	 GetPreConditionMatrix(&PreConditionMatrix,Size);

	//Allocating Memory for HVector on Host
	HVector = (float *)malloc_safe_call(Size * sizeof(float));
	
	//HVector = PreConditionMatrix Inverse * ResidueVector
	SolvePreConditionMatrix(PreConditionMatrix,HVector,ResidueVector,Size);
	
	//Allocating Memory on Device Memory
	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceHVector,Size*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceDirectionVector,Size*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceDelta0,sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceTempValue,sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceTempMatrix,Size*Size*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceDelta1,sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceTempBuffer,Size*sizeof(float)) );

	//Copying Hvector from Host to Device
	cudaMemcpy((void *)DeviceHVector,(void *)HVector,Size*sizeof(float),cudaMemcpyHostToDevice);

	//Intialize DirectionVector i.e DirectionVector = -(HVector)
	IntializeDirectionVector(&DirectionVector,Size,DeviceHVector,DeviceDirectionVector,DimGrid,DimBlock);

	//Allocating memory for Delat0
	Delta0 = (float *)malloc_safe_call(sizeof(float));
        	
	//Compute Delta0, Delta0 = ResidueVector * HVector
	VectorVectorDotProduct<<<DimGrid,DimBlock>>>(DeviceResidueVector,DeviceHVector,DeviceTempBuffer,DeviceDelta0,Size,BLOCKSIZE);
	
	//Copy Value of Delta0 from Device to Host
        cudaMemcpy((void *)Delta0,(void *)DeviceDelta0,sizeof(float),cudaMemcpyDeviceToHost);
	
	if(*Delta0 < EPSILON)
                 exit(-1);
	
	//Allocating Memory for Delta1
	Delta1 = (float *)malloc_safe_call(sizeof(float));
		
	//Allocating Memory for Temporary Variable to hold Intermediate Value
	TempValue = (float *)malloc_safe_call(sizeof(float));
        	
	PresentIteration = 0;
	do
	{
		//Incrementing the Iteration Count
		PresentIteration++;

		//Compute Tau, Tau = Delta0/DirectionVectorTranspose*CoeffientMatrix*DirectionVector
		//Computing First DirectionVectorTranpose * CoeffientMatrix
		
		//Invoking Vector Matrix Multiplication Kernel
		VectorMatrixMultiplication<<<DimGrid,DimBlock>>>(DeviceCoefficientMatrix,DeviceDirectionVector,DeviceTempBuffer,Size,BLOCKSIZE);
		
		//Invoking Vector Vector Dot Product Kernel
		VectorVectorDotProduct<<<DimGrid,DimBlock>>>(DeviceTempBuffer,DeviceDirectionVector,DeviceTempBuffer,DeviceTempValue,Size,BLOCKSIZE);	
				
		//Copying the TempValue from the Device to Host
		cudaMemcpy((void *)TempValue,(void *)DeviceTempValue,sizeof(float),cudaMemcpyDeviceToHost);
		
		Tau = (*Delta0) / (*TempValue);
		
		//Compute New Solution Vector, NewSolutionVector = OldSolutionVector + Tau*DirectionVector
		ScalarVectorMultiplication<<<DimGrid,DimBlock>>>(DeviceDirectionVector,Tau,DeviceTempBuffer,Size,BLOCKSIZE);
		VectorVectorAddition<<<DimGrid,DimBlock>>>(DeviceSolutionVector,DeviceTempBuffer,Size,BLOCKSIZE);
		
		//Copying the solution from Device to Host
		cudaMemcpy((void *)SolutionVector,(void *)DeviceSolutionVector,Size*sizeof(float),cudaMemcpyDeviceToHost);
		
		//Compute New ResidueVector, NewResidueVector = OldResidueVector - Tau*CoefficientMatrix*DirectionVector
		ScalarMatrixMultiplication<<<DimGrid,DimBlock>>>(DeviceCoefficientMatrix,Tau,DeviceTempMatrix,Size,BLOCKSIZE);
		MatrixVectorMultiplication<<<DimGrid,DimBlock>>>(DeviceTempMatrix,DeviceDirectionVector,DeviceTempBuffer,Size,BLOCKSIZE);
		VectorVectorAddition<<<DimGrid,DimBlock>>>(DeviceResidueVector,DeviceTempBuffer,Size,BLOCKSIZE);
		
		//Copying ResidueVector From Device to Host
		cudaMemcpy((void *)ResidueVector,(void *)DeviceResidueVector,Size*sizeof(float),cudaMemcpyDeviceToHost);
	
		//Computing the new Delta value i.e Delta1 = DotProduct of ResidueVector and HVector
		
		//Updating the HVector
		SolvePreConditionMatrix(PreConditionMatrix,HVector,ResidueVector,Size);
		
		//Copying the new HVector from Host to Device
		cudaMemcpy((void*)DeviceHVector,(void *)HVector,Size*sizeof(float),cudaMemcpyHostToDevice);
		
		VectorVectorDotProduct<<<DimGrid,DimBlock>>>(DeviceResidueVector,DeviceHVector,DeviceTempBuffer,DeviceDelta1,Size,BLOCKSIZE); 
		
		//Copying the value of Delta1 from Device to Host
		cudaMemcpy((void *)Delta1,(void *)DeviceDelta1,sizeof(float),cudaMemcpyDeviceToHost);
		
		if(*Delta1 < EPSILON)
			break;
		Beta = (*Delta1) / (*Delta0);
		*Delta0 = *Delta1;
		
		//Computing new Direction Vector,NewDirectionVector = -HVector + Beta*OldDirectionVector
		ScalarVectorMultiplication<<<DimGrid,DimBlock>>>(DeviceDirectionVector,Beta,DeviceTempBuffer,Size,BLOCKSIZE);
		VectorVectorSubtraction<<<DimGrid,DimBlock>>>(DeviceTempBuffer,DeviceHVector,DeviceDirectionVector,Size,BLOCKSIZE);
		
		//copying New Direction vector From Device to Host
		cudaMemcpy((void *)DirectionVector,(void *)DeviceDirectionVector,Size*sizeof(float),cudaMemcpyDeviceToHost);

	}while(*Delta0 > EPSILON && PresentIteration < MaxIterations);

	//Stop timing the computation
	gettimeofday(&tv,NULL);
   	double t2=tv.tv_sec+(tv.tv_usec/1000000.0);

	//Calculating total time taken for computation
	cputime = t2 - t1;
	
	//Freeing the Memory allocated in Device
	FreeTheDeviceMemory(DeviceCoefficientMatrix,DeviceRhsVector,DeviceSolutionVector,DeviceDirectionVector,DeviceHVector,DeviceResidueVector,DeviceTempMatrix,DeviceDelta0,DeviceDelta1,DeviceTempValue,DeviceTempBuffer);

	//Printing the Solution
	for(RowNum=0;RowNum<Size;RowNum++)
        {
        	printf("%f",SolutionVector[RowNum]);
                printf("\t");
        }
        printf("\n");

	//printing the timing and the number of iterations
        printf("Solution Vector with %d size, displayed above is calculated in %d iterations and in %lf seconds \n",Size,PresentIteration,cputime);

	//Freeing the Memory allocated in Host
	FreeTheHostMemory(CoefficientMatrix,RhsVector,SolutionVector,DirectionVector,HVector,ResidueVector,Delta0,Delta1,TempValue);

	return(0);
}//End of main
//--------------------------------------------------------------------------------------------------------------------------------------------
void GetPreConditionMatrix(float **PreConditionMatrix,int Size)
{
	float *TempPreConditionMatrix;
	int RowNum,ColNum,Index = 0;

	//Allocating Memory for Precondition Matrix
	TempPreConditionMatrix = (float *)malloc_safe_call(Size * Size * sizeof(float));
	
	//Filling the Precondition Matrix i.e Identity Matrix with ones and zeros
	for(RowNum =0;RowNum<Size;RowNum++)
	{
		for(ColNum=0;ColNum<Size;ColNum++)
		{
			if(RowNum == ColNum)
				TempPreConditionMatrix[Index++] = 1.0;
			else
				TempPreConditionMatrix[Index++] = 0.0;
		}
	}
	*PreConditionMatrix = TempPreConditionMatrix;
}
//---------------------------------------------------------------------------------------------------------------------
void SolvePreConditionMatrix(float *PreConditionMatrix,float *HVector,float *ResidueVector,int Size)
{
	int RowNum;
	//HVector = PreConditionMatrix Inverse * ResidueVector
	//Since PreConditionMatrix is an Identity Matrix or Unit Matrix and Inverse of an Identity Matrix is Itself
	//AI = IA = A, HVector is Residue Vector itself

	//Assigning ResidueVector to HVector
	for(RowNum =0;RowNum<Size;RowNum++)
		HVector[RowNum] = ResidueVector[RowNum];
}
//------------------------------------------------------------------------------------------------------------------------
void GetTheInputToTheProblem(float **RhsVector,float **SolutionVector,float **CoefficientMatrix,int Size)
{
	float *TempRhsVector,*TempSolutionVector,*TempCoefficientArray;
	float **TempCoefficientMatrix;
	int RowNum;
	TempRhsVector = (float *)malloc_safe_call(Size * sizeof(float));
	
	TempSolutionVector = (float *)malloc_safe_call(Size * sizeof(float));
	
	TempCoefficientArray = (float *)malloc_safe_call(Size * Size * sizeof(float));
	
	TempCoefficientMatrix = (float **)malloc_safe_call(Size * sizeof(float *));
	for(RowNum=0;RowNum<Size;RowNum++)
		TempCoefficientMatrix[RowNum] = (float *)malloc_safe_call(Size * sizeof(float));
	
	//Filling Solution Vector with zeroes i.e Intial Solution Vector
	for(RowNum =0;RowNum<Size;RowNum++)
		TempSolutionVector[RowNum] = 0.0;
	
	//Generating the Coefficient Matrix i.e Symmetric matrix which is Positive Definite
	GenerateCoefficientMatrix(TempCoefficientMatrix,Size);

	//Converting Coefficient Matrix to One dimensional Array
	ConvertMatrixToOneDimension(TempCoefficientMatrix,TempCoefficientArray,Size);

	//Generating the Rhs vector
	GenerateRhsVector(TempCoefficientMatrix,TempRhsVector,Size);
	
	//Assigning Temporary Variables to Original Variables
	*RhsVector = TempRhsVector;
	*SolutionVector = TempSolutionVector;
	*CoefficientMatrix = TempCoefficientArray;

	//Freeing the memory allocated for TempCoefficientMatrix
	for(RowNum=0;RowNum<Size;RowNum++)
	      free(TempCoefficientMatrix[RowNum]);
	free(TempCoefficientMatrix);
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------------
void  CalculateResidueVector(float **ResidueVector,float *DeviceCoefficientMatrix,float *DeviceSolutionVector,float *DeviceTempBuffer,int Size,float *DeviceRhsVector,float *DeviceResidueVector,dim3 DimGrid,dim3 DimBlock)
{
	float *TempResidueVector;
	TempResidueVector = (float *)malloc_safe_call(Size * sizeof(float));
        
        //Multiplying Coefficient Matrix and the Intial solution Vector
        MatrixVectorMultiplication<<<DimGrid,DimBlock>>>(DeviceCoefficientMatrix,DeviceSolutionVector,DeviceTempBuffer,Size,BLOCKSIZE)    ;
  
        //Subracting Result of AX and the Rhs Vector
        VectorVectorSubtraction<<<DimGrid,DimBlock>>>(DeviceTempBuffer,DeviceRhsVector,DeviceResidueVector,Size,BLOCKSIZE);
  
        //copying Residue vector from Device to Host
        cudaMemcpy((void *)TempResidueVector,(void *)DeviceResidueVector,Size*sizeof(float),cudaMemcpyDeviceToHost);
	
	*ResidueVector = TempResidueVector;
}
//-----------------------------------------------------------------------------------------------------------------------------------------
void IntializeDirectionVector(float **DirectionVector,int Size,float *DeviceHVector,float *DeviceDirectionVector,dim3 DimGrid,dim3 DimBlock)
{
	float *TempDirectionVector;
	TempDirectionVector = (float *)malloc_safe_call(Size * sizeof(float));
        
        ScalarVectorSubtraction<<<DimGrid,DimBlock>>>(DeviceDirectionVector,DeviceHVector,0.0,Size,BLOCKSIZE);
	//Copying Direction Vector from Device to Host
 	cudaMemcpy((void *)TempDirectionVector,(void *)DeviceDirectionVector,Size*sizeof(float),cudaMemcpyDeviceToHost);
	
	*DirectionVector = TempDirectionVector;
}
//--------------------------------------------------------------------------------------------------------------------------------
void GenerateCoefficientMatrix(float **CoefficientMatrix,int Size)
{
    int RowNum,ColNum;
    float TempValue = 0.0;
    for(RowNum = 0;RowNum<Size;RowNum++)
    {
      for(ColNum = 0;ColNum<Size;ColNum++)
	CoefficientMatrix[RowNum][ColNum] = RowNum + ColNum;
    }
    //Making the Coefficient matrix as diagonal dominant by Adding rows elements and putting in the corresponding Diagonal element
    for(RowNum=0;RowNum<Size;RowNum++)
    {
      TempValue = 0.0;
      for(ColNum=0;ColNum<Size;ColNum++)
	TempValue += CoefficientMatrix[RowNum][ColNum];
      CoefficientMatrix[RowNum][RowNum] = TempValue;
    }
}
//-----------------------------------------------------------------------------------------------------------------------------------------
void ConvertMatrixToOneDimension(float **CoefficientMatrix,float *CoefficientArray,int Size)
{
  int Index,RowNum,ColNum;
  Index = 0;
  for(RowNum = 0;RowNum<Size;RowNum++)
  {
    for(ColNum = 0;ColNum<Size;ColNum++)
	CoefficientArray[Index++] = CoefficientMatrix[RowNum][ColNum];
  }
}
//-------------------------------------------------------------------------------------------------------------------------------------------
void GenerateRhsVector(float **CoefficientMatrix,float *RhsVector,int Size)
{
  int RowNum,ColNum,Index = 0;
  float TempValue;
  for(RowNum=0;RowNum<Size;RowNum++)
  {
    TempValue = 0.0;
    for(ColNum=0;ColNum<Size;ColNum++)
	  TempValue += CoefficientMatrix[RowNum][ColNum];
    RhsVector[Index++] = TempValue;
  }
}
//---------------------------------------------------------------------------------------------------------------------------------------------
unsigned long long int rdtsc(void)
{
   unsigned long long int x;

   __asm__ volatile(".byte 0x0f,0x31" : "=A" (x));
   return x;
}
//---------------------------------------------------------------------------------------------------------------------------------------------------
void FreeTheDeviceMemory(float *DeviceCoefficientMatrix,float *DeviceRhsVector,float *DeviceSolutionVector,float *DeviceDirectionVector,float *DeviceHVector,float *DeviceResidueVector,float *DeviceTempMatrix,float *DeviceDelta0,float *DeviceDelta1,float *DeviceTempValue,float *DeviceTempBuffer)
{
	CUDA_SAFE_CALL( cudaFree(DeviceCoefficientMatrix) );
	CUDA_SAFE_CALL( cudaFree(DeviceRhsVector) );
	CUDA_SAFE_CALL( cudaFree(DeviceSolutionVector) );
	CUDA_SAFE_CALL( cudaFree(DeviceDirectionVector) );
	CUDA_SAFE_CALL( cudaFree(DeviceHVector) );
	CUDA_SAFE_CALL( cudaFree(DeviceResidueVector) );
	CUDA_SAFE_CALL( cudaFree(DeviceTempMatrix) );
	CUDA_SAFE_CALL( cudaFree(DeviceDelta0) );
	CUDA_SAFE_CALL( cudaFree(DeviceDelta1) );
	CUDA_SAFE_CALL( cudaFree(DeviceTempValue) );
	CUDA_SAFE_CALL( cudaFree(DeviceTempBuffer) );
}
//----------------------------------------------------------------------------------------------------------------------------------------------------
void  FreeTheHostMemory(float *CoefficientMatrix,float *RhsVector,float *SolutionVector,float *DirectionVector,float *HVector,float *ResidueVector,float *Delta0,float *Delta1,float *TempValue)
{
	free(CoefficientMatrix);
	free(RhsVector);
	free(SolutionVector);
	free(DirectionVector);
	free(HVector);
	free(ResidueVector);
	free(Delta0);
	free(Delta1);
	free(TempValue);
}
//-----------------------------------------------------------------------------------------------------------------------------------------------------
void *malloc_safe_call(int size)
{
	void *ptr;
	
	ptr = malloc(size);
	
	if(ptr==NULL)
	{
		printf("memory unavailable\n");
		exit(-1);
	}	
	
	return(ptr);	
}
//--------------------------------------------------------------------------------------------------------------------------
