/*****************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example     : cuda-matrix-matrix-multiplication-cublas-mgpu.cu
 
  Objective   : Write a CUDA Program for Matrix Matrix multiplication using 
                CUBLAS3 library function calls to be executed on multiple GPUs.

  Input       : None 

  Output      : Execution time in seconds , Gflops achieved
                                                                                                                            
  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

****************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>

#include "cublas.h"

#define EPS 1.0e-12
#define GRIDSIZE 10
#define BLOCKSIZE 16

#define SIZE 128

typedef struct
{
        int hA;
        int wA;
        int wB;
        double* hMatA;
        double* hMatB;
        double* hMatC; 
        double* dMatA;
        double* dMatB;
        double* dMatC;
        cudaStream_t stream;
}TGPUPlan; 

cudaEvent_t *start,*stop;
float elapsedTime;
float Tsec;
cudaDeviceProp deviceProp;

int size=SIZE;

void  CUBLAS_SAFE_CALL(cublasStatus status)                                  
{
        if(status != CUBLAS_STATUS_SUCCESS)                     
         { printf(" Error in CUBLAS call.Program terminating\n");
            exit(-1);                                           
         }                                      
}

int Max_GPU =2;

/* Check for safe return of all calls to the device */
void CUDA_SAFE_CALL(cudaError_t call)
{
        cudaError_t ret = call;
        //printf("RETURN FROM THE CUDA CALL:%d\t:",ret);
        switch(ret)
        {
                case cudaSuccess:
                //              printf("Success\n");
                                break;
        /*      case cudaErrorInvalidValue:
                                {
                                printf("ERROR: InvalidValue:%i.\n",__LINE__);
                                exit(-1);
                                break;
                                }
                case cudaErrorInvalidDevicePointer:
                                {
                                printf("ERROR:Invalid Device pointeri:%i.\n",__LINE__);
                                exit(-1);
                                break;
                                }
                case cudaErrorInvalidMemcpyDirection:
                                {
                                printf("ERROR:Invalid memcpy direction:%i.\n",__LINE__);
                                exit(-1);
                                break;
                                }                       */
                default:
                        {
                                printf(" ERROR at line :%i.%d' ' %s\n",__LINE__,ret,cudaGetErrorString(ret));
                                exit(-1);
                                break;
                        }
        }
}

/* Compare CPU and GPU results */
void relError(double* dRes,double* hRes,int size)
{
        double relativeError=0.0,errorNorm=0.0;
        int flag=0;
        int i;

        for( i = 0; i < size; ++i) {
                if (fabs(hRes[i]) > fabs(dRes[i]))
                        relativeError = fabs((hRes[i] - dRes[i]) / hRes[i]);
                else
                        relativeError = fabs((dRes[i] - hRes[i]) / dRes[i]);

                if (relativeError > EPS && relativeError != 0.0e+00 )
                {
                        if(errorNorm < relativeError)
                        {
                                errorNorm = relativeError;
                                flag=1;
                        }
                }

        }
        if( flag == 1)
        {
                printf(" \n Results verfication : Failed");
                printf(" \n Considered machine precision : %e", EPS);
                printf(" \n Relative Error                  : %e\n", errorNorm);

        }
        else
                printf("\n Results verfication : Success\n");

}

/* prints the result on screen */
void print_on_screen(char * program_name,float tsec,double gflops,int size,int flag)//flag=1 if gflops has been calculated else flag =0
{
        printf("\n---------------%s----------------\n",program_name);
        printf("\tSIZE\t TIME_SEC\t Gflops\n");
        if(flag==1)
        printf("\t%d\t%f\t%lf\t",size,tsec,gflops);
        else
        printf("\t%d\t%lf\t%lf\t",size,"---","---");

}

/* Print memory error */
void mem_error(char *arrayname, char *benchmark, int len, char *type)
{

        printf("\nMemory not sufficient to allocate for array %s\n\tBenchmark : %s  \n\tMemory requested = %d number of %s elements\n",arrayname, benchmark, len, type);
       printf("\tAborting\n");
       exit(-1);
}

/* Routine for verifiying the Multi-GPU results against the CPU results */
void checkResult(double *InMatA, double *InMatB, double *outMatC, int m, int n , int k )
{	printf(" checking results...\n");
        int     i;
        int     j;
        int     k1;
        double  *tempOut;

        tempOut  = (double*) malloc (m * n * sizeof(double));
        if (tempOut == 0){
        printf("\n Memory allocation Failed for Resultant Matrix");
        exit (0);
        }

 	for (i = 0; i < m  ; ++i) {
                for (j = 0; j < n; ++j) {
                        double  cprod = 0;
                        for (k1 = 0; k1 < k; ++k1)
                                cprod += InMatA[k1*m+i] * InMatB[j *k + k1];
                tempOut[i*n+j] = cprod;//alpha * cprod + beta * tempOut[j * n + i];
		
                }
        }

        printf("\n..............\n");

        /*** check relative error with approx precision ****/
	relError(outMatC,tempOut,m*n);

        free(tempOut);
}
/* Function to calculate Gflops */
double calculate_gflops(float &Tsec)
{
//        printf("time taken is %.8lf\n",Tsec);
        float gflops=(1.0e-9 * (( 2.0 * size*size )/Tsec));
  //      printf("Gflops is \t%f\n",gflops);
        return gflops;
}

/* Fill in the vector with double precision values */
void fill_dp_vector(double* vec,int size)
{
        int ind;
        for(ind=0;ind<size;ind++)
                vec[ind]=drand48();
}



int main(int argc,char** argv)
{
	int numGPU;
	int hA,wA,wB;
	int i;

	double *host_A,*host_B,*host_C;	

	hA=wA=wB=SIZE;
	printf("%d,%d,%d\n",hA,wA,wB);
	int lda,ldb,ldc;
	lda=hA;
	ldb=wA;
	ldc=wB;

	/* ----- MULTI DEVICE COUNT --------*/
	CUDA_SAFE_CALL(cudaGetDeviceCount(&numGPU));
	if(numGPU > Max_GPU )
		numGPU=Max_GPU;
	printf("CUDA CAPABLE DEVICE COUNT: %d\n",numGPU);	


	/* allocate memory for GPU events */
        start = (cudaEvent_t *) malloc (sizeof(cudaEvent_t)* numGPU);
        stop = (cudaEvent_t *) malloc (sizeof(cudaEvent_t) *numGPU);

	if(start==NULL)
                mem_error("start","matmatmul_mGPU_cublas",1,"cudaEvent_t");
        if(stop==NULL)
                mem_error("stop","matmatmul_mGPU_cublas",1,"cudaEvent_t");




	/*---------FILLING HOST MATRICES---------*/
	
	host_A = (double *) malloc (hA*wA*sizeof(double));
	host_B=(double *)malloc(wA*wB*sizeof(double));
	host_C=(double *)malloc(hA*wB*sizeof(double));

	/* Initializing the input arrays */
        fill_dp_vector(host_A,hA*wA);
        fill_dp_vector(host_B,wA*wB);





	/*-------INITIATING THE DATA FOR EACH DEVICE ----*/
	TGPUPlan plan[numGPU];
	for(i =0;i < numGPU; i++)
	{
		plan[i].hA = hA;
		plan[i].wA = wA;
		plan[i].wB = wB/numGPU;

	}

	for(i = 0;i < wB % numGPU; i++)		
		plan[i].wB++;

	for(i = 0; i<numGPU ; i++)	
	{	printf("\n for Device %d, Matrix dimensions are %d %d %d \n",i,plan[i].hA,plan[i].wA,plan[i].wB);
		plan[i].hMatA=(double*)malloc(plan[i].hA*plan[i].wA*sizeof(double));
		plan[i].hMatB=(double*)malloc(plan[i].wA*plan[i].wB*sizeof(double));
		plan[i].hMatC=(double*)malloc(plan[i].hA*plan[i].wB*sizeof(double));
	}

	int gpuBase=0;
	for(i =0;i < numGPU ;i++)
	{
		plan[i].hMatA = host_A ;
		plan[i].hMatB = host_B+gpuBase;
		gpuBase += plan[i].wA * plan[i].wB ; 	
	
/*	int j;printf(" Matrix A..\n");
	for(j=0;j<hA*wA;j++)
	printf(" %lf ",plan[i].hMatA[j]);
	printf(" Matrix B..\n");
	for(j=0;j<hA*plan[i].wB;j++)
        printf(" %lf ",plan[i].hMatB[j]);
	printf("\n.............\n");
*/	}


	for( i=0; i<numGPU ;i++)
	{
		CUDA_SAFE_CALL(cudaSetDevice(i));
		CUDA_SAFE_CALL(cudaStreamCreate(&plan[i].stream));
		CUDA_SAFE_CALL(cudaEventCreate(&start[i]));
	}
	for( i=0; i<numGPU ;i++)
	{
		CUDA_SAFE_CALL(cudaSetDevice(i));
		printf("Device %d is set....\n",i);
		CUBLAS_SAFE_CALL(cublasAlloc (hA*wA, sizeof(double), (void**)&plan[i].dMatA));
      		CUBLAS_SAFE_CALL(cublasAlloc (wA*plan[i].wB, sizeof(double), (void**)&plan[i].dMatB));
		CUBLAS_SAFE_CALL(cublasAlloc (hA*plan[i].wB, sizeof(double), (void**)&plan[i].dMatC));
	
		 // Initialization of vectors with host vectors 

              	CUBLAS_SAFE_CALL(cublasSetVector (hA*wA, sizeof(double), plan[i].hMatA, 1,plan[i].dMatA, 1));
   		CUBLAS_SAFE_CALL(cublasSetVector (wA*plan[i].wB, sizeof(double), plan[i].hMatB, 1,plan[i].dMatB, 1));

		double alpha=1.0;
		double beta=0.0;

		cublasDgemm ('N', 'N', hA , plan[i].wB , wA  , alpha, plan[i].dMatA, lda, plan[i].dMatB, ldb , beta,plan[i].dMatC, ldc );

		CUBLAS_SAFE_CALL(cublasGetVector (hA*plan[i].wB, sizeof(double), plan[i].dMatC, 1,plan[i].hMatC, 1));

	}


	printf("\n...........................................\n");


	int  offset=0;
        for(i=0; i<numGPU ; i++)
        {       printf(" \nDevice %d\n",i);
         	CUDA_SAFE_CALL(cudaSetDevice(i));
		int device;
		cudaGetDevice(&device);
	        cudaGetDeviceProperties(&deviceProp,device);
	        printf("Using device %d: %s \n", device, deviceProp.name);
	

		/* printing the result on screen */
	    	print_on_screen("MULTI_GPU CUBLAS MAT MAT MULTIPLICATION",Tsec,calculate_gflops(Tsec),size,0);




		/**** Combining The Result Matrices to Single Matrix  ************/

		int j;
                for( j=0;j < plan[i].hA*plan[i].wB ; j++ )
                        host_C[(j%hA)*wB+j/hA+offset] = plan[i].hMatC[j];
                offset += plan[i].wB ;
	}

	checkResult(host_A,host_B,host_C,hA,wB,wA);

	return 0;
}

