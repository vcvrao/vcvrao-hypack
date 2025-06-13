/************************************************************************************************************************************
    			C-DAC Tech Workshop : hyPACK-2013
                             October 15-18,2012

 Example 1.1           : gpgpu_perf.cu

 Objective             : Performance of Matrix Matrix Multiplication  
 
 Description	       : Provided are 3 functions to show Matrix Matrix Multiplicaiton performance on GPU's. Each function exploits
			 various hardware features of GPU's to gain performance. One can notice performance will double each time as 
			 one goes from executing from function 1 to 3.   
			 Features that are exploited:
			 1)Block Size 2)Thread Mapping 3)Shared Memory 4)Global Memory Bandwidth 5)Registers 6)Scheduling 7)Tiling

			 Lower <function numbers> may not exploit all these features (or to a lesser degree) but as the <function number> 
			 increases features will be exploited more agressively.  
 
			 Function Name: matMulBlockwise<function number> - function that performs matrix matrix multiplication
			 
 Input                 : Set <Square Matrix Size> <Shared Memory Size> <GPGPU Device Number> <Function Number>
 			 1) <Shared Memory Size> can only take 16, 32, 48 as values 
			 2) <Function Number> can only take 1, 2, 3 as values

 Output                : Time taken and gflops for Matrix Matrix Multiplication in individual function runs based on <Function Number>.                                                  
 Created               : August 2013 
       
 E-mail                : hpcfte@cdac.in      

************************************************************************************************************************************/


#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define RANGEA 100
#define RANGEB 500

struct matrixData
{
	int row, col;
	int blockARowSz, blockAColSz;
	int blockBRowSz, blockBColSz;
	int subBlockRowSz, subBlockColSz;
	int alignment;
	int funcNum;
};

struct performanceMeasure
{
	double gflops;
        double opCount;
	float time;
	cudaEvent_t start, stop;
};

struct kernelExecutionConfig
{
	dim3 gridDim, blockDim;
	unsigned int dynamicSharedMemSz;
 	int gpgpuDeviceNum;
        cudaDeviceProp gpuProperties;
};

__host__ void callMatMulBlockwise(double ** __restrict__ matrixA, double ** __restrict__ matrixB, double ** __restrict__ resultMatrix, double ** __restrict__ matrixA_D, double ** __restrict__ matrixB_D, double ** __restrict__ resultMatrix_D, struct matrixData * mData, struct matrixData ** mData_D, char ** argv);

__global__ void matMulBlockwise1(double * __restrict__ matrixA, double * __restrict__ matrixB, double * __restrict__ resultMatrix, struct matrixData * mData);
__global__ void matMulBlockwise2(double * __restrict__ matrixA, double * __restrict__ matrixB, double * __restrict__ resultMatrix, struct matrixData * mData);
__global__ void matMulBlockwise3(double * __restrict__ matrixA, double * __restrict__ matrixB, double * __restrict__ resultMatrix, struct matrixData * mData);

__host__ void setupMatrixResource(double ** matrixA, double ** matrixB, double ** resultMatrix, double ** matrixA_D, double ** matrixB_D, double ** resultMatrix_D, struct matrixData * mdata, struct kernelExecutionConfig *  kerExecConf);
__host__ void releaseMatrixResource(double ** matrixA, double ** matrixB, double ** resultMatrix, double ** matrixA_D, double ** matrixB_D, double ** resultMatrix_D);
__host__ void startTime(struct performanceMeasure * perfM);
__host__ void end_Time(struct performanceMeasure * perfM);
__host__ void setKernelData(struct kernelExecutionConfig * kerExecConf, struct matrixData * mData);

__host__ int generateMatrixData(double * matrix, unsigned int row, unsigned int col, unsigned int range, int blockRSz, int blockCSz, int flag);
__host__ void displayMatrix(double * matrix, int row, int col);
__host__ void writeToFile(double * resultMatrix, unsigned int row, unsigned int col, char * fileName);
__host__ void allocateMatrixMemory1D(double ** matrix, int row, int col);
__host__ void freeHeapSpace1D(double ** matrix);
__host__ void simpleMatrixMultiplication(double * MatrixA,  int rowASize, int colASize, double * MatrixB, int colBSize, double * resultMatrix);

int main(int argc, char ** argv)
{
	if(argc!=5)
	{
		printf("Usage:executable.out matrixSz sharedMemSz(Kb)=(16, 32 or 48) gpgpuDeveiceNum funcNum=(1, 2 or 3)\n");
		printf("./a.out 1024 48 0 1\n");
		printf("./a.out 1024 16 0 2\n");
		printf("./a.out 1024 16 0 3\n");
		exit(EXIT_FAILURE);
	}

	struct matrixData mData;
	mData.row=atoi(argv[1]);
	mData.col=atoi(argv[1]);
	mData.blockARowSz=64;
	mData.blockAColSz=64;
	mData.blockBRowSz=64;
	mData.blockBColSz=64;
	mData.alignment=64;
	mData.funcNum=atoi(argv[4]);

	if(mData.funcNum!=1 && mData.funcNum!=2 && mData.funcNum!=3)
	{
		printf("Usage: funcNum can only take 1, 2 or 3 as values\n");
		exit(EXIT_FAILURE);
	}

	double * matrixA=NULL, * matrixB=NULL, *resultMatrix=NULL;
	double * matrixA_D=NULL, * matrixB_D=NULL, *resultMatrix_D=NULL;
	struct matrixData * mData_D=NULL;


	callMatMulBlockwise(&matrixA, &matrixB, &resultMatrix, &matrixA_D, &matrixB_D, &resultMatrix_D, &mData, &mData_D, argv);

	cudaMemcpy(resultMatrix, resultMatrix_D, mData.row*mData.col*sizeof(double), cudaMemcpyDeviceToHost);
        //displayMatrix(resultMatrix, mData.row, mData.col);
        writeToFile(resultMatrix, mData.row, mData.col, "gpu.r");


	double * r=NULL;
	allocateMatrixMemory1D(&r, mData.row, mData.col);
	generateMatrixData(r, mData.row, mData.col, RANGEA, mData.blockARowSz, mData.blockBColSz, 0);
        simpleMatrixMultiplication(matrixA,  mData.row, mData.col, matrixB, mData.col, r);
        writeToFile(r, mData.row, mData.col, "cpu.r");

	int k=0;
	for(int i=0; i<mData.row*mData.col; ++i)
	{
		double m=resultMatrix[i]-r[i];
		if(m<0)
			m*=-1.0F;
		if(m>0.000001F)
		{
			k++;
			//printf("%lf %lf %lf %d\n", resultMatrix[i], r[i], m, i);
		}
	}
	printf("Data Miss Matches with sequential multiplication is %d\n", k);

	free(r);

	releaseMatrixResource(&matrixA, &matrixB, &resultMatrix, &matrixA_D, &matrixB_D, &resultMatrix_D);
	return 0;
}

__host__ void setKernelData(struct kernelExecutionConfig * kerExecConf, struct matrixData * mData)
{
        kerExecConf->gridDim.x=mData->col/mData->blockBColSz;
        kerExecConf->gridDim.y=mData->row/mData->blockBRowSz;
        kerExecConf->blockDim.x=mData->subBlockColSz;
        kerExecConf->blockDim.y=mData->subBlockRowSz;
	if(kerExecConf->gpuProperties.sharedMemPerBlock<kerExecConf->dynamicSharedMemSz)
	{
		printf("Dynamic shared memory size should be less than 48Kb\n");
		exit(EXIT_FAILURE);
	}
}

__global__ void matMulBlockwise3(double * __restrict__ matrixA, double * __restrict__ matrixB, double * __restrict__ resultMatrix, struct matrixData * mData)
{
	extern __shared__ double smem[] __align__(16);
	double R[16], tempb[4], tempa[4];
	int i=0, ii=0, l=0;
	const int itrSz=mData->row*(1.0F/mData->blockARowSz);
	const int subBlockSz=mData->subBlockRowSz*mData->subBlockColSz;	

	for(i=0; i<16; ++i)
	{
		R[i]=0.0F;
	}

	for(i=0; i<itrSz; ++i)
	{
		#pragma unroll 2 
		for(ii=0; ii<4; ++ii)
		{
			smem[(4*subBlockSz)+(threadIdx.y*mData->subBlockColSz+threadIdx.x)]=matrixB[(i*mData->blockAColSz*mData->col)+(blockIdx.x*mData->blockAColSz)+(ii*mData->subBlockRowSz*mData->col)+(0*mData->subBlockColSz)+(threadIdx.y*mData->col+threadIdx.x)];
			smem[(5*subBlockSz)+(threadIdx.y*mData->subBlockColSz+threadIdx.x)]=matrixB[(i*mData->blockAColSz*mData->col)+(blockIdx.x*mData->blockAColSz)+(ii*mData->subBlockRowSz*mData->col)+(1*mData->subBlockColSz)+(threadIdx.y*mData->col+threadIdx.x)];
			smem[(6*subBlockSz)+(threadIdx.y*mData->subBlockColSz+threadIdx.x)]=matrixB[(i*mData->blockAColSz*mData->col)+(blockIdx.x*mData->blockAColSz)+(ii*mData->subBlockRowSz*mData->col)+(2*mData->subBlockColSz)+(threadIdx.y*mData->col+threadIdx.x)];
			smem[(7*subBlockSz)+(threadIdx.y*mData->subBlockColSz+threadIdx.x)]=matrixB[(i*mData->blockAColSz*mData->col)+(blockIdx.x*mData->blockAColSz)+(ii*mData->subBlockRowSz*mData->col)+(3*mData->subBlockColSz)+(threadIdx.y*mData->col+threadIdx.x)];
		
			smem[(0*subBlockSz)+(threadIdx.y*mData->subBlockColSz+threadIdx.x)]=matrixA[(blockIdx.y*mData->blockARowSz*mData->col)+(i*mData->blockAColSz)+(0*mData->subBlockRowSz*mData->col)+(ii*mData->subBlockColSz)+(threadIdx.y*mData->col+threadIdx.x)];
			smem[(1*subBlockSz)+(threadIdx.y*mData->subBlockColSz+threadIdx.x)]=matrixA[(blockIdx.y*mData->blockARowSz*mData->col)+(i*mData->blockAColSz)+(1*mData->subBlockRowSz*mData->col)+(ii*mData->subBlockColSz)+(threadIdx.y*mData->col+threadIdx.x)];
			smem[(2*subBlockSz)+(threadIdx.y*mData->subBlockColSz+threadIdx.x)]=matrixA[(blockIdx.y*mData->blockARowSz*mData->col)+(i*mData->blockAColSz)+(2*mData->subBlockRowSz*mData->col)+(ii*mData->subBlockColSz)+(threadIdx.y*mData->col+threadIdx.x)];
			smem[(3*subBlockSz)+(threadIdx.y*mData->subBlockColSz+threadIdx.x)]=matrixA[(blockIdx.y*mData->blockARowSz*mData->col)+(i*mData->blockAColSz)+(3*mData->subBlockRowSz*mData->col)+(ii*mData->subBlockColSz)+(threadIdx.y*mData->col+threadIdx.x)];

			__syncthreads();

			#pragma unroll 16
			for(l=0; l<16; ++l)
			{
				tempb[0]=smem[(4*subBlockSz)+(l*mData->subBlockColSz+threadIdx.x)];
				tempb[1]=smem[(5*subBlockSz)+(l*mData->subBlockColSz+threadIdx.x)];
				tempb[2]=smem[(6*subBlockSz)+(l*mData->subBlockColSz+threadIdx.x)];
				tempb[3]=smem[(7*subBlockSz)+(l*mData->subBlockColSz+threadIdx.x)];

				tempa[0]=smem[(0*subBlockSz)+(threadIdx.y*mData->subBlockColSz)+l];
				tempa[1]=smem[(1*subBlockSz)+(threadIdx.y*mData->subBlockColSz)+l];
				tempa[2]=smem[(2*subBlockSz)+(threadIdx.y*mData->subBlockColSz)+l];
				tempa[3]=smem[(3*subBlockSz)+(threadIdx.y*mData->subBlockColSz)+l];
				
				R[0]+=tempa[0]*tempb[0];
				R[1]+=tempa[0]*tempb[1];
				R[2]+=tempa[0]*tempb[2];
				R[3]+=tempa[0]*tempb[3];
				
				R[4]+=tempa[1]*tempb[0];
				R[5]+=tempa[1]*tempb[1];
				R[6]+=tempa[1]*tempb[2];
				R[7]+=tempa[1]*tempb[3];
				
				R[8]+=tempa[2]*tempb[0];
				R[9]+=tempa[2]*tempb[1];
				R[10]+=tempa[2]*tempb[2];
				R[11]+=tempa[2]*tempb[3];

				R[12]+=tempa[3]*tempb[0];
				R[13]+=tempa[3]*tempb[1];
				R[14]+=tempa[3]*tempb[2];
				R[15]+=tempa[3]*tempb[3];
			}
			__syncthreads();

		}
	}

	
	for(ii=0; ii<4; ++ii)
	{
		resultMatrix[(blockIdx.y*mData->blockARowSz*mData->col)+(blockIdx.x*mData->blockAColSz)+(ii*mData->subBlockRowSz*mData->col)+(0*mData->subBlockColSz)+(threadIdx.y*mData->col+threadIdx.x)]=R[4*ii];
		resultMatrix[(blockIdx.y*mData->blockARowSz*mData->col)+(blockIdx.x*mData->blockAColSz)+(ii*mData->subBlockRowSz*mData->col)+(1*mData->subBlockColSz)+(threadIdx.y*mData->col+threadIdx.x)]=R[4*ii+1];
		resultMatrix[(blockIdx.y*mData->blockARowSz*mData->col)+(blockIdx.x*mData->blockAColSz)+(ii*mData->subBlockRowSz*mData->col)+(2*mData->subBlockColSz)+(threadIdx.y*mData->col+threadIdx.x)]=R[4*ii+2];
		resultMatrix[(blockIdx.y*mData->blockARowSz*mData->col)+(blockIdx.x*mData->blockAColSz)+(ii*mData->subBlockRowSz*mData->col)+(3*mData->subBlockColSz)+(threadIdx.y*mData->col+threadIdx.x)]=R[4*ii+3];
	}

}

__global__ void matMulBlockwise2(double * __restrict__ matrixA, double * __restrict__ matrixB, double * __restrict__ resultMatrix, struct matrixData * mData)
{
	extern __shared__ double smem[] __align__(16);
	const int blockRowSz=mData->blockBRowSz; 
	const int blockColSz=mData->blockBColSz;
	const int row=mData->row;
	const int col=mData->col;
	const int subBlockRowSz=mData->subBlockRowSz; 
	const int subBlockColSz=mData->subBlockColSz;
	const int subBlockSz=subBlockRowSz*subBlockColSz;
	int i=0, j=0, l=0;
	const int itrSz=row/blockRowSz;
	double r[8], tempb, tempb1;

	for(i=0; i<8; ++i)
	{
		r[i]=0.0F;
	}

	for(i=0; i<itrSz; ++i)
	{
		#pragma unroll 6
		for(j=0; j<=6; j+=2)
		{
        		smem[(0*subBlockSz)+(threadIdx.y*subBlockColSz+threadIdx.x)]=matrixB[(i*blockRowSz*col)+(blockIdx.x*blockColSz)+(j*subBlockRowSz*col)+(threadIdx.y*col+threadIdx.x)];
        		smem[(subBlockSz)+(threadIdx.y*subBlockColSz+threadIdx.x)]=matrixB[(i*blockRowSz*col)+(blockIdx.x*blockColSz)+((j+1)*subBlockRowSz*col)+(threadIdx.y*col+threadIdx.x)];
        		smem[(2*subBlockSz)+(threadIdx.y*subBlockColSz+threadIdx.x)]=matrixA[(blockIdx.y*blockRowSz*col)+(i*blockColSz)+(j*subBlockRowSz)+(threadIdx.x*col+threadIdx.y)];
        		smem[(3*subBlockSz)+(threadIdx.y*subBlockColSz+threadIdx.x)]=matrixA[(blockIdx.y*blockRowSz*col)+(i*blockColSz)+((j+1)*subBlockRowSz)+(threadIdx.x*col+threadIdx.y)];	
			__syncthreads();
			
			#pragma unroll 4
			for(l=0; l<8; ++l)
			{
				tempb=smem[(0*subBlockSz)+(l*subBlockColSz)+threadIdx.x];
				tempb1=smem[(subBlockSz)+(l*subBlockColSz)+threadIdx.x];
				r[0]+=smem[(2*subBlockSz)+(l*subBlockColSz)+(threadIdx.y)]*tempb;
				r[1]+=smem[(2*subBlockSz)+(l*subBlockColSz)+(8+threadIdx.y)]*tempb;
				r[2]+=smem[(2*subBlockSz)+(l*subBlockColSz)+(16+threadIdx.y)]*tempb;
				r[3]+=smem[(2*subBlockSz)+(l*subBlockColSz)+(24+threadIdx.y)]*tempb;
				r[4]+=smem[(2*subBlockSz)+(l*subBlockColSz)+(32+threadIdx.y)]*tempb;
				r[5]+=smem[(2*subBlockSz)+(l*subBlockColSz)+(40+threadIdx.y)]*tempb;
				r[6]+=smem[(2*subBlockSz)+(l*subBlockColSz)+(48+threadIdx.y)]*tempb;
				r[7]+=smem[(2*subBlockSz)+(l*subBlockColSz)+(56+threadIdx.y)]*tempb;
				
				r[0]+=smem[(3*subBlockSz)+(l*subBlockColSz)+(threadIdx.y)]*tempb1;
				r[1]+=smem[(3*subBlockSz)+(l*subBlockColSz)+(8+threadIdx.y)]*tempb1;
				r[2]+=smem[(3*subBlockSz)+(l*subBlockColSz)+(16+threadIdx.y)]*tempb1;
				r[3]+=smem[(3*subBlockSz)+(l*subBlockColSz)+(24+threadIdx.y)]*tempb1;
				r[4]+=smem[(3*subBlockSz)+(l*subBlockColSz)+(32+threadIdx.y)]*tempb1;
				r[5]+=smem[(3*subBlockSz)+(l*subBlockColSz)+(40+threadIdx.y)]*tempb1;
				r[6]+=smem[(3*subBlockSz)+(l*subBlockColSz)+(48+threadIdx.y)]*tempb1;
				r[7]+=smem[(3*subBlockSz)+(l*subBlockColSz)+(56+threadIdx.y)]*tempb1;
			}
			__syncthreads();
		}
	}

	for(j=0; j<8; ++j)
	{
		resultMatrix[(blockIdx.y*blockRowSz*col)+(blockIdx.x*blockColSz)+(j*subBlockRowSz*col)+(threadIdx.y*col+threadIdx.x)]=r[j];
	}
}

__global__ void matMulBlockwise1(double * __restrict__ matrixA, double * __restrict__ matrixB, double * __restrict__ resultMatrix, struct matrixData * mData)
{
	extern __shared__ double smem[] __align__(16);
	int blockRowSz=mData->blockBRowSz; //block dim
	int blockColSz=mData->blockBColSz;
	int row=mData->row;
	int col=mData->col;
	int subBlockRowSz=mData->subBlockRowSz; 
	int subBlockColSz=mData->subBlockColSz;
	int subBlockSz=subBlockRowSz*subBlockColSz;
	int subItrSz=blockRowSz/subBlockRowSz;
	int blockSz=blockRowSz*blockColSz;
	int i=0, j=0, l=0;
	int itrSz=row/blockRowSz;
	double result;
	int disp=blockSz+threadIdx.y*subBlockColSz;


	for(i=0; i<itrSz; ++i)
	{
		for(j=0; j<subItrSz; ++j)
		{
        		smem[(j*subBlockSz)+(threadIdx.y*subBlockColSz+threadIdx.x)]=matrixB[(i*blockRowSz*col)+(blockIdx.x*blockColSz)+(j*subBlockRowSz*col)+(threadIdx.y*col+threadIdx.x)];
		}

		for(j=0; j<subItrSz; ++j)
		{
        		smem[blockSz+(threadIdx.y*subBlockColSz+threadIdx.x)]=matrixA[(blockIdx.y*blockRowSz*col)+(i*blockColSz)+(j*subBlockRowSz*col)+(threadIdx.y*col+threadIdx.x)];
			result=0.0F;
			__syncthreads();
 
			#pragma unroll 16  
			for(l=0; l<subBlockColSz; ++l)
			{
				result=fma(smem[disp+l], smem[threadIdx.x+(l)*subBlockColSz], result);
				/*
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+1)]*smem[threadIdx.x+(l+1)*subBlockColSz];
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+2)]*smem[threadIdx.x+(l+2)*subBlockColSz];
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+3)]*smem[threadIdx.x+(l+3)*subBlockColSz];
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+4)]*smem[threadIdx.x+(l+4)*subBlockColSz];
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+5)]*smem[threadIdx.x+(l+5)*subBlockColSz];
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+6)]*smem[threadIdx.x+(l+6)*subBlockColSz];
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+7)]*smem[threadIdx.x+(l+7)*subBlockColSz];
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+8)]*smem[threadIdx.x+(l+8)*subBlockColSz];
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+9)]*smem[threadIdx.x+(l+9)*subBlockColSz];
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+10)]*smem[threadIdx.x+(l+10)*subBlockColSz];
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+11)]*smem[threadIdx.x+(l+11)*subBlockColSz];
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+12)]*smem[threadIdx.x+(l+12)*subBlockColSz];
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+13)]*smem[threadIdx.x+(l+13)*subBlockColSz];
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+14)]*smem[threadIdx.x+(l+14)*subBlockColSz];
				result+=smem[blockSz+(threadIdx.y*subBlockColSz+l+15)]*smem[threadIdx.x+(l+15)*subBlockColSz];
				*/
			}
			resultMatrix[(blockIdx.y*blockRowSz*col)+(blockIdx.x*blockColSz)+(j*subBlockRowSz*col)+(threadIdx.y*col+threadIdx.x)]+=result;
		}
		__syncthreads();
	}
}

__host__ void callMatMulBlockwise(double ** __restrict__ matrixA, double ** __restrict__ matrixB, double ** __restrict__ resultMatrix, double ** __restrict__ matrixA_D, double ** __restrict__ matrixB_D, double ** __restrict__ resultMatrix_D, struct matrixData * mData, struct matrixData ** mData_D, char ** argv)
{
	//Set up time measurement
	struct kernelExecutionConfig kerExecConf;
	kerExecConf.dynamicSharedMemSz=atoi(argv[2])*1024;

	if(atoi(argv[2])!=16 && atoi(argv[2])!=32 && atoi(argv[2])!=48)
	{
		printf("Usage: sharedMemSz can only take 16, 32 or 48 as values\n");
		exit(EXIT_FAILURE);
	}

	kerExecConf.gpgpuDeviceNum=atoi(argv[3]);
	cudaGetDeviceProperties(&(kerExecConf.gpuProperties), kerExecConf.gpgpuDeviceNum);
	setupMatrixResource(matrixA, matrixB, resultMatrix, matrixA_D, matrixB_D, resultMatrix_D, mData, &kerExecConf);
	struct performanceMeasure perfM;
	perfM.opCount=2.0F*mData->row*mData->col*mData->col*1E-9;
	printf("GPU Share Mem Matrix Multiplication\n");

	if(mData->funcNum==1)
	{
		mData->subBlockRowSz=8;
		mData->subBlockColSz=64;
	}
	else if(mData->funcNum==2)
	{
		mData->subBlockRowSz=8;
		mData->subBlockColSz=64;
	}
	else
	{
		mData->subBlockRowSz=16;
		mData->subBlockColSz=16;
	}

	cudaMalloc((void **)mData_D, sizeof(struct matrixData));
        cudaMemcpy(*mData_D, mData, sizeof(struct matrixData), cudaMemcpyHostToDevice);

	startTime(&perfM);
	
	setKernelData(&kerExecConf, mData);

	if(mData->funcNum==1)
	{
		matMulBlockwise1<<<kerExecConf.gridDim, kerExecConf.blockDim, kerExecConf.dynamicSharedMemSz>>>(*matrixA_D, *matrixB_D, *resultMatrix_D, *mData_D);
	}
	else if(mData->funcNum==2)
	{
		matMulBlockwise2<<<kerExecConf.gridDim, kerExecConf.blockDim, kerExecConf.dynamicSharedMemSz>>>(*matrixA_D, *matrixB_D, *resultMatrix_D, *mData_D);
	}
	else if(mData->funcNum==3)
	{
		matMulBlockwise3<<<kerExecConf.gridDim, kerExecConf.blockDim, kerExecConf.dynamicSharedMemSz>>>(*matrixA_D, *matrixB_D, *resultMatrix_D, *mData_D);
	}

	end_Time(&perfM);
       
	printf("GPU Share Mem Matrix Multiplication Done\n");
        printf("%.6lf seconds elapsed\n", perfM.time*1E-3);
        perfM.gflops=perfM.opCount*(1.0F/perfM.time)*1E3;
        printf("%.6lf gflops\n", perfM.gflops);
	cudaError_t error=cudaGetLastError();
	printf("%s\n", cudaGetErrorString (error));
        cudaFree(*mData_D);
	*mData_D=NULL;
}

__host__ void startTime(struct performanceMeasure * perfM)
{
        perfM->time=0.0f;
        cudaEventCreate (&(perfM->start));
        cudaEventCreate (&(perfM->stop));
        cudaEventRecord (perfM->start, 0);
}

__host__ void end_Time(struct performanceMeasure * perfM)
{
        cudaEventRecord (perfM->stop, 0);
        cudaEventSynchronize (perfM->stop);
        cudaEventElapsedTime (&(perfM->time), perfM->start, perfM->stop);
        cudaEventDestroy (perfM->start);
        cudaEventDestroy (perfM->stop);
}

__host__ void setupMatrixResource(double ** matrixA, double ** matrixB, double ** resultMatrix, double ** matrixA_D, double ** matrixB_D, double ** resultMatrix_D, struct matrixData * mData, struct kernelExecutionConfig * kerExecConf)
{
	allocateMatrixMemory1D(matrixA, mData->row, mData->col);
	allocateMatrixMemory1D(matrixB, mData->row, mData->col);
	allocateMatrixMemory1D(resultMatrix, mData->row, mData->col);
	
	unsigned int sz=mData->row*mData->col*sizeof(double);
	unsigned int temp=2*sz;
	if(temp>kerExecConf->gpuProperties.totalGlobalMem)
	{
		printf("Total GPU Global memory size should is %u\n", kerExecConf->gpuProperties.totalGlobalMem);
		exit(EXIT_FAILURE);
	}
	cudaMalloc((void **)matrixA_D, sz);
        cudaMalloc((void **)matrixB_D, sz);
        cudaMalloc((void **)resultMatrix_D, sz);

        generateMatrixData(*matrixA, mData->row, mData->col, RANGEA, mData->blockARowSz, mData->blockAColSz, 1);
        generateMatrixData(*matrixB, mData->row, mData->col, RANGEB, mData->blockBRowSz, mData->blockBColSz, 1);
        generateMatrixData(*resultMatrix, mData->row, mData->col, RANGEA, mData->blockARowSz, mData->blockBColSz, 0);

        cudaMemcpy(*matrixA_D, *matrixA, sz, cudaMemcpyHostToDevice);
        cudaMemcpy(*matrixB_D, *matrixB, sz, cudaMemcpyHostToDevice);
        cudaMemcpy(*resultMatrix_D, *resultMatrix, sz, cudaMemcpyHostToDevice);
}

__host__ void releaseMatrixResource(double ** matrixA, double ** matrixB, double ** resultMatrix, double ** matrixA_D, double ** matrixB_D, double ** resultMatrix_D)
{
	freeHeapSpace1D(matrixA);
	freeHeapSpace1D(matrixB);
	freeHeapSpace1D(resultMatrix);

	cudaFree(*matrixA_D);
        cudaFree(*matrixB_D);
        cudaFree(*resultMatrix_D);
	*matrixA_D=*matrixB_D=*resultMatrix_D=NULL;
}

__host__ void allocateMatrixMemory1D(double ** matrix, int row, int col)
{
	int size=row*col*sizeof(double);
 	#ifdef ICC
       		*matrix=_mm_maloc(size, alignment);
        #elif GCC
        	if(posix_memalign((void **)&(*matrix), aligment, size))
        	{
        	        exit(EXIT_FAILURE);
        	}
        #else
        	*matrix=(double *)malloc(size);
        #endif
	if(!*matrix)
	{	
		exit(EXIT_FAILURE);
	}
}

__host__ void freeHeapSpace1D(double ** matrix)
{
 	#ifdef ICC
        	_mm_free(*matrix);
        #else
		free(*matrix);
        #endif

	*matrix=NULL;
}

__host__ int generateMatrixData(double * matrix, unsigned int row, unsigned int col, unsigned int range, int blockRSz, int blockCSz, int flag)
{
        if(!matrix)
                return 1;

        srand((unsigned)time(NULL));
        double randomNumber=0.0;

	for(int i=0; i<row*col; i++)
	{
		if(flag)
		{
			randomNumber = rand() / (RAND_MAX + 1.0);
			//randomNumber = i+1.0F;
			matrix[i]=randomNumber*range;
		}
		else
                	matrix[i]=0.0F;
	}

        return 0;
}


__host__ void writeToFile(double * resultMatrix, unsigned int row, unsigned int col, char * fileName)
{
        //Output of Simple Matrix Multiplication written to this file
        if(!fileName)
        {

                printf("Invalid File Name.\n");
                return;
        }

        FILE * resultFile=fopen(fileName, "w");
        if(!resultFile)
        {
                printf("Output File for storing Matrix Multiplication result, could not be open.\n");
                return;
        }

        int i=0;

        for(i=0; i<row*col; i++)
        {
                if(i!=0 && i%col==0)
                {
                        fprintf(resultFile, "\n%f\t", resultMatrix[i]);
                }
                else
                {
                        fprintf(resultFile, "%f\t", resultMatrix[i]);
                }
        }
        fclose(resultFile);
}


__host__ void displayMatrix(double * matrix, int row, int col)
{
        int i=0;

        for(i=0; i<row*col; i++)
        {
                if(i!=0 && i%col==0)
                {
                        printf("\n%f\t", matrix[i]);
                }
                else
                {
                        printf("%f\t", matrix[i]);
                }
        }
        printf("\n");
}

//Simple Matrix Multiplication
__host__ void simpleMatrixMultiplication(double * MatrixA,  int rowASize, int colASize, double * MatrixB, int colBSize, double * resultMatrix)
{
        //Setting up time measurement for calculating matrix multiplication
        struct timeval tim;
        printf("Performing Simple Matrix Multiplication\n");
        gettimeofday(&tim, NULL);
        double startTime=tim.tv_sec+(tim.tv_usec*1E-6), gflops=0.0F, diffTime=0.0F;
        double opCount=2.0F*rowASize*colASize*colBSize*1E-9;

        int i=0, j=0, k=0, l=0;
        double result=0.0;

        for(i=0; i<rowASize; i++)
        {
                for(j=0; j<colBSize; j++)
                {
                        result=0.0;
                        for(k=0; k<colASize; k++)
                        {
                                result+=MatrixA[colASize*i+k]*MatrixB[j+k*colBSize];
                        }
                        resultMatrix[l++]=result;
                }
        }

        gettimeofday(&tim, NULL);
        printf("Simple Matrix Multiplication Done\n");
        double endTime=tim.tv_sec+(tim.tv_usec*1E-6);
        diffTime=endTime-startTime;
        printf("%.6lf seconds elapsed\n\n", diffTime);
        gflops=opCount*(1.0F/diffTime);
        printf("%.6lf gflops\n\n", gflops);
}

