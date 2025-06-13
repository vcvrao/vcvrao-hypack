/****************************************************************************

	C-DAC Tech Workshop : hyPACK-2013
                October 15-18, 2013

  Objective   : program to solve a  linear system of equations (Ax = b) using conjugate 
                gradient method in a GPU (using cublas lib)

  Input       : Number of unknowns and max no. of iterations

  Output      : Solution Vector.
	                                                                                                                    
  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

****************************************************************************/
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<cuda.h>
#include "cublas.h"
//----------------------------------------------------------------------------------------------------
#define BLOCKSIZE 16
#define EPSILON 1.0E-20
//----------------------------------------------------------------------------------------------------
void GetTheInputToTheProblem(float **,float **,float **,int );
void GetPreConditionMatrix(float ** ,int );
void SolvePreConditionMatrix(float *,float *,float *,int );
void GenerateCoefficientMatrix(float **,int);
void ConvertMatrixToOneDimension(float **,float *,int );
void GenerateRhsVector(float **,float *,int ); 
void IntializeDirectionVector(float ** ,int ,float * ,float * ,dim3 ,dim3 );
void CalculateResidueVector(float ** ,float * ,float * ,float * ,int ,    float * ,float * ,dim3 ,dim3 );
void FreeTheDeviceMemory(float *,float *,float *,float *,float *,float *,float *);
void FreeTheHostMemory(float *,float *,float *,float *,float *,float *);
void *malloc_safe_call(int size);
//--------------------------------------------------------------------------------------------------------

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

int main(int argc,char **argv)
{
	//Checking if  Valid number of Arguements have been passed
	if(argc != 3)
	{
		printf("Usage:<./a.out><No. of unknowns><Maximum No. of Iterations> \n");
		exit(-1);
	}

	// Variable Declaration
	float *RhsVector,*SolutionVector,*CoefficientMatrix,*ResidueVector;
	float *PreConditionMatrix,*HVector,*DirectionVector;
	float *DeviceRhsVector,*DeviceSolutionVector,*DeviceCoefficientMatrix,*DeviceResidueVector;
	float *DeviceHVector,*DeviceDirectionVector;
	int    MaxIterations,Size,PresentIteration,rowNum;
	float Delta0, TempValue, *DeviceTempBuffer;
	float Delta1;
	float Tau,Beta = 0.0;
	struct timeval tv;
	double timing;	

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

	//intialising cublas 	
	cublasInit();
	
	//start timing the computation
	gettimeofday(&tv, NULL);
   	double t1=tv.tv_sec+(tv.tv_usec/1000000.0);

	//Calculate the Residue vector RESIDUE=AX-B
	CalculateResidueVector(&ResidueVector,DeviceCoefficientMatrix,DeviceSolutionVector,DeviceTempBuffer,Size,DeviceRhsVector,DeviceResidueVector,DimGrid,DimBlock);
	
	//Getting Pre-Conditional Matrix, Preconditional Matrix is Identity Matrix	
	 GetPreConditionMatrix(&PreConditionMatrix,Size);

	//Allocating Memory for HVector on Host
	HVector = (float *)malloc_safe_call(Size * sizeof(float));
	
	//HVector = PreConditionMatrix Inverse * ResidueVector
	SolvePreConditionMatrix(PreConditionMatrix,HVector,ResidueVector,Size);
	
	//Allocating Memory on Device Memory
	CUDA_SAFE_CALL(cudaMalloc( (void **)&DeviceHVector,Size*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc( (void **)&DeviceDirectionVector,Size*sizeof(float)));
	
	//Copying Hvector from Host to Device
	cudaMemcpy((void *)DeviceHVector,(void *)HVector,Size*sizeof(float),cudaMemcpyHostToDevice);

	//Intialize DirectionVector i.e DirectionVector = -(HVector)
	IntializeDirectionVector(&DirectionVector,Size,DeviceHVector,DeviceDirectionVector,DimGrid,DimBlock);

	//Compute Delta0 = ResidueVector * HVector
	Delta0 = cublasSdot (Size, DeviceResidueVector, 1, DeviceHVector, 1);

	if(Delta0 < EPSILON)
                 exit(-1);
					
	PresentIteration = 0;
	do
	{
		//Incrementing the Iteration Count
		PresentIteration++;

		//Compute Tau, Tau = Delta0/DirectionVectorTranspose*CoeffientMatrix*DirectionVector
		//Computing First DirectionVectorTranpose * CoeffientMatrix
		
		cublasSgemv('T', Size, Size, 1, DeviceCoefficientMatrix, Size, DeviceDirectionVector, 1, 0, DeviceTempBuffer, 1);
		
		//multiplying the above result with direction vector
		TempValue = cublasSdot (Size, DeviceTempBuffer, 1, DeviceDirectionVector, 1);
			
		Tau = Delta0 / TempValue;
		
		//Compute New Solution Vector, NewSolutionVector = OldSolutionVector + Tau*DirectionVector
		cublasSaxpy (Size, Tau, DeviceDirectionVector, 1, DeviceSolutionVector, 1);
		
		//Copying the solution from Device to Host
		cudaMemcpy((void *)SolutionVector,(void *)DeviceSolutionVector,Size*sizeof(float),cudaMemcpyDeviceToHost);
		
		//Compute New ResidueVector, NewResidueVector = OldResidueVector + Tau*CoefficientMatrix*DirectionVector
		cublasSgemv('N', Size, Size, 1, DeviceCoefficientMatrix, Size, DeviceDirectionVector, 1, 0, DeviceTempBuffer, 1);
		cublasSaxpy (Size, Tau, DeviceTempBuffer, 1, DeviceResidueVector, 1);
		
		//Copying ResidueVector From Device to Host
		cudaMemcpy((void *)ResidueVector,(void *)DeviceResidueVector,Size*sizeof(float),cudaMemcpyDeviceToHost);
	
		//Computing the new Delta value i.e Delta1 = DotProduct of ResidueVector and HVector
		
		//Updating the HVector
		SolvePreConditionMatrix(PreConditionMatrix,HVector,ResidueVector,Size);
		
		//Copying the new HVector from Host to Device
		cudaMemcpy((void*)DeviceHVector,(void *)HVector,Size*sizeof(float),cudaMemcpyHostToDevice);
		
		Delta1 = cublasSdot (Size, DeviceResidueVector, 1, DeviceHVector, 1);
		
		if(Delta1 < EPSILON)
			break;
			
		Beta = Delta1 / Delta0;
		Delta0 = Delta1;
		
		//Computing new Direction Vector,NewDirectionVector = -HVector + Beta*OldDirectionVector
		cublasSscal (Size, Beta, DeviceDirectionVector, 1);
		cublasSaxpy (Size, -1, DeviceHVector, 1, DeviceDirectionVector, 1);
		
				
	}while(Delta0 > EPSILON && PresentIteration < MaxIterations);
	
	//Stop timing the computation
	gettimeofday(&tv,NULL);
   	double t2=tv.tv_sec+(tv.tv_usec/1000000.0);

	//Calculating total time taken for computation
	timing = t2 - t1;	

	//shutting down cublas
	cublasShutdown();

	//Freeing the Memory allocated in Device
	FreeTheDeviceMemory(DeviceCoefficientMatrix,DeviceRhsVector,DeviceSolutionVector,DeviceDirectionVector,DeviceHVector,DeviceResidueVector,DeviceTempBuffer);

	//Printing the Solution
	for(rowNum=0;rowNum<Size;rowNum++)
        {
        	printf("%f",SolutionVector[rowNum]);
                printf("\t");
        }
        printf("\n");

	//printing the timing and the number of iterations
        printf("Solution Vector with %d size, displayed above is calculated in %d iterations and in %lf seconds\n",Size,PresentIteration,timing);

	//Freeing the Memory allocated in Host
	FreeTheHostMemory(CoefficientMatrix,RhsVector,SolutionVector,DirectionVector,HVector,ResidueVector);

	return(0);
}//End of main
//--------------------------------------------------------------------------------------------------------------------------------------------
void GetPreConditionMatrix(float **PreConditionMatrix,int Size)
{
	float *TempPreConditionMatrix;
	int rowNum,colNum,Index = 0;

	//Allocating Memory for Precondition Matrix
	TempPreConditionMatrix = (float *)malloc_safe_call(Size * Size * sizeof(float));
	
	//Filling the Precondition Matrix i.e Identity Matrix with ones and zeros
	for(rowNum =0;rowNum<Size;rowNum++)
	{
		for(colNum=0;colNum<Size;colNum++)
		{
			if(rowNum == colNum)
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
	int rowNum;
	//HVector = PreConditionMatrix Inverse * ResidueVector
	//Since PreConditionMatrix is an Identity Matrix or Unit Matrix and Inverse of an Identity Matrix is Itself
	//AI = IA = A, HVector is Residue Vector itself

	//Assigning ResidueVector to HVector
	for(rowNum =0;rowNum<Size;rowNum++)
		HVector[rowNum] = ResidueVector[rowNum];
}
//------------------------------------------------------------------------------------------------------------------------
void GetTheInputToTheProblem(float **RhsVector,float **SolutionVector,float **CoefficientMatrix,int Size)
{
	float *TempRhsVector,*TempSolutionVector,*TempCoefficientArray;
	float **TempCoefficientMatrix;
	int rowNum;
	TempRhsVector = (float *)malloc_safe_call(Size * sizeof(float));
	
	TempSolutionVector = (float *)malloc_safe_call(Size * sizeof(float));
	
	TempCoefficientArray = (float *)malloc_safe_call(Size * Size * sizeof(float));
	
	TempCoefficientMatrix = (float **)malloc_safe_call(Size * sizeof(float *));
	for(rowNum=0;rowNum<Size;rowNum++)
		TempCoefficientMatrix[rowNum] = (float *)malloc_safe_call(Size * sizeof(float));
	
	//Filling Solution Vector with zeroes i.e Intial Solution Vector
	for(rowNum =0;rowNum<Size;rowNum++)
		TempSolutionVector[rowNum] = 0.0;
	
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
	for(rowNum=0;rowNum<Size;rowNum++)
	      free(TempCoefficientMatrix[rowNum]);
	free(TempCoefficientMatrix);
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------------
void  CalculateResidueVector(float **ResidueVector,float *DeviceCoefficientMatrix,float *DeviceSolutionVector,float *DeviceTempBuffer,int Size,float *DeviceRhsVector,float *DeviceResidueVector,dim3 DimGrid,dim3 DimBlock)
{
	float *TempResidueVector;
	TempResidueVector = (float *)malloc_safe_call(Size * sizeof(float));
   
        //Multiplying Coefficient Matrix and the Intial solution Vector and subtracting the result from the RHS vector
        cublasScopy(Size, DeviceRhsVector, 1, DeviceTempBuffer, 1);

        cublasSgemv ('N', Size, Size, 1, DeviceCoefficientMatrix, Size, DeviceSolutionVector, 1, -1, DeviceTempBuffer, 1);

        cublasScopy(Size, DeviceTempBuffer, 1, DeviceResidueVector, 1);
  
        //copying Residue vector from Device to Host
        cudaMemcpy((void *)TempResidueVector,(void *)DeviceResidueVector,Size*sizeof(float),cudaMemcpyDeviceToHost);
	
	*ResidueVector = TempResidueVector;
}
//-----------------------------------------------------------------------------------------------------------------------------------------
void IntializeDirectionVector(float **DirectionVector,int Size,float *DeviceHVector,float *DeviceDirectionVector,dim3 DimGrid,dim3 DimBlock)
{
	float *TempDirectionVector;
	TempDirectionVector = (float *)malloc_safe_call(Size * sizeof(float));
               
       	//DeviceDirectionVector = -DeviceHVector
        cublasScopy(Size, DeviceHVector, 1, DeviceDirectionVector, 1);
        cublasSscal (Size, -1, DeviceDirectionVector, 1);
        
	//Copying Direction Vector from Device to Host
 	cudaMemcpy((void *)TempDirectionVector,(void *)DeviceDirectionVector,Size*sizeof(float),cudaMemcpyDeviceToHost);
	
	*DirectionVector = TempDirectionVector;
}
//--------------------------------------------------------------------------------------------------------------------------------
void GenerateCoefficientMatrix(float **CoefficientMatrix,int Size)
{
    int rowNum,colNum;
    float TempValue = 0.0;
    for(rowNum = 0;rowNum<Size;rowNum++)
    {
      for(colNum = 0;colNum<Size;colNum++)
	CoefficientMatrix[rowNum][colNum] = rowNum + colNum;
    }
    //Making the Coefficient matrix as diagonal dominant by Adding rows elements and putting in the corresponding Diagonal element
    for(rowNum=0;rowNum<Size;rowNum++)
    {
      TempValue = 0.0;
      for(colNum=0;colNum<Size;colNum++)
	TempValue += CoefficientMatrix[rowNum][colNum];
      CoefficientMatrix[rowNum][rowNum] = TempValue;
    }
}
//-----------------------------------------------------------------------------------------------------------------------------------------
void ConvertMatrixToOneDimension(float **CoefficientMatrix,float *CoefficientArray,int Size)
{
  int Index,rowNum,colNum;
  Index = 0;
  for(rowNum = 0;rowNum<Size;rowNum++)
  {
    for(colNum = 0;colNum<Size;colNum++)
	CoefficientArray[Index++] = CoefficientMatrix[rowNum][colNum];
  }
}
//-------------------------------------------------------------------------------------------------------------------------------------------
void GenerateRhsVector(float **CoefficientMatrix,float *RhsVector,int Size)
{
  int rowNum,colNum,Index = 0;
  float TempValue;
  for(rowNum=0;rowNum<Size;rowNum++)
  {
    TempValue = 0.0;
    for(colNum=0;colNum<Size;colNum++)
	  TempValue += CoefficientMatrix[rowNum][colNum];
    RhsVector[Index++] = TempValue;
  }
}
//---------------------------------------------------------------------------------------------------------------------------------------------------
void FreeTheDeviceMemory(float *DeviceCoefficientMatrix,float *DeviceRhsVector,float *DeviceSolutionVector,float *DeviceDirectionVector,float *DeviceHVector,float *DeviceResidueVector, float *DeviceTempBuffer)
{
	cudaFree(DeviceCoefficientMatrix);
	cudaFree(DeviceRhsVector);
	cudaFree(DeviceSolutionVector);
	cudaFree(DeviceDirectionVector);
	cudaFree(DeviceHVector);
	cudaFree(DeviceResidueVector);
	cudaFree(DeviceTempBuffer);
}
//----------------------------------------------------------------------------------------------------------------------------------------------------
void  FreeTheHostMemory(float *CoefficientMatrix,float *RhsVector,float *SolutionVector,float *DirectionVector,float *HVector,float *ResidueVector)
{
	free(CoefficientMatrix);
	free(RhsVector);
	free(SolutionVector);
	free(DirectionVector);
	free(HVector);
	free(ResidueVector);
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
