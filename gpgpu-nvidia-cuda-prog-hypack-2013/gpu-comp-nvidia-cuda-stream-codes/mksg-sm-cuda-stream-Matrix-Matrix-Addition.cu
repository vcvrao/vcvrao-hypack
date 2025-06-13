
/*****************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example    : mksg-sm-cuda-stream-Matrix-Matrix-Addition.cu 
              (Multi-kernel Single GPU Stream Matrix-Matrix-Addition) 
 
  Objective  : The objective is to demonstrate use of CUDA Synchronous 
               and CUDA  Asynchronous APIs with CUDA streams for simple
               addition of two square matrices & compare the execution time 
               Matrix-Matrix Addition kernel is domonstrated 

  Input      : Can specify the number of kernels. If not specified it will 
               be taken as default 16.

  Output     : Execution-Type(Syn,Asyn),Execution Time in sec Relative-Error
  
  Created    : August-2013

  E-mail     : hpcfte@cdac.in     

****************************************************************************/

/*  inclusion of header file that contains necessary declarions */
#include <stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<time.h>
#include<math.h>
#include<assert.h>
#define EPS 1.0e-14 /* threshhold aprrox epsilion value */
#define TOTALEVENT 5  /* Toltal number of GPU event to be recorded */
#define WIDTH 64
#define HEIGHT 128
#define BLOCK_SIZE 8


/* initialize host martices*/
double *hAddMatMatA , *hAddMatMatB, *hAddMatMatC;  // holds host matrices
double   *dAddMatMatA, *dAddMatMatB, *dAddMatMatC; // holds device matrices
int nkernels;                   // holds total number of concurrent kernels


/* function prototypes */
double matMatAddCheckResult (double *hAddMatMatA,double *hAddMatMatB,double *output,int row, int col);
void memoryAlloc(int ,int ,int , int );



/* Macro to check for correctness of CUDA API */
#define CUDA_SAFE_CALL(call){\
	cudaError_t err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(-1);                                                  \
    }}\


/*
 * Fill in the matrix/vector with double precision values
 */
void fillInData(double* vec,int size)
{
        int ind;
        for(ind=0;ind<size;ind++)
                vec[ind]=drand48() ;
}

/*
*check mem error
*/
void memError(char *arrayname, char *benchmark, int len, char *type)
{

        printf("\nMemory not sufficient to allocate for array %s\n\tBenchmark : %s  \n\tMemory requested = %d number of %s elements\n",arrayname, benchmark, len, type);
        printf("\n\tAborting\n\n");
        exit(-1);
}

/*
*check grid and block dimensions
*/
void checkBlockGridDim(cudaDeviceProp devProp,dim3 blockDim,dim3 gridDim)
{

        if( blockDim.x >= devProp.maxThreadsDim[0] || blockDim.y >= devProp.maxThreadsDim[1] || blockDim.z >= devProp.maxThreadsDim[2] )
        {
                printf("\nBlock Dimensions exceed the maximum limits:%d * %d * %d \n",devProp.maxThreadsDim[0],devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
               exit(-1);
        }

        if( gridDim.x >= devProp.maxGridSize[0] || gridDim.y >= devProp.maxGridSize[1] || gridDim.z >= devProp.maxGridSize[2] )
        {
                printf("\nGrid Dimensions exceed the maximum limits:%d * %d * %d \n",devProp.maxGridSize[0],devProp.maxGridSize[1],devProp.maxGridSize[2]);
               exit(-1);
        }
}


/*****************************************
* Matrix Matrix Addition
******************************************/
__global__ void kernelMatMatAdd(double *dInMatA, double *dInMatB,double *dInMatC,  int matRowSize,int matColSize, int threadDim)
  {
	int tidx = threadIdx.x;
    	int tidy = threadIdx.y;
   	int tindex = (threadDim * tidx) + tidy;
    	int maxNumThread = threadDim * threadDim;
    	int pass = 0;
    	int rowCount ;
    	int curColInd ;

    	while( (curColInd = (tindex + maxNumThread * pass))  < matColSize )
     	{
        	for( rowCount = 0; rowCount < matRowSize; rowCount++)
		{
          		 dInMatC[curColInd * matRowSize + rowCount] = dInMatA[curColInd * matRowSize + rowCount] + dInMatB[curColInd * matRowSize + rowCount];
		}
        	pass++;       // move to next column
      	} 

     	__syncthreads();

  }/* end of Mat Mat Add device code */



/***************************************************************
function to implement concurrent kernel execution 
***************************************************************/
void funcAsynchConcurrentExec(double *dAddMatMatA, double *dAddMatMatB, double *dAddMatMatC,double *hAddMatMatA, double *hAddMatMatB, double *hAddMatMatC, int nkernels, int NSTREAM, cudaStream_t *stream , int hA, int wA, int hB, int wB, cudaDeviceProp deviceProp)
{
    	float elapsedTime;           // holds timing variables
    	cudaError_t err;               // holds error value

    	/* create CUDA event handles */

	cudaEvent_t startEvent, stopEvent;
	CUDA_SAFE_CALL( cudaEventCreate(&startEvent));
        CUDA_SAFE_CALL( cudaEventCreate(&stopEvent));

	
	/* get all errors before kernel launch */
        if ( err=cudaGetLastError())
        {
                printf(" File : %s , Line : %d , Error : %s \n",__FILE__, __LINE__, cudaGetErrorString(err));
        }

        /* define blocks and grids check grid and block dimension*/
        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE); //threads per block
        dim3 dimGrid(1,1); //blocks per grid
        checkBlockGridDim(deviceProp,dimGrid, dimBlock);

        /* Asynchronous kernel execution */
        
	cudaEventRecord(startEvent);
	for( int ind=0; ind<nkernels; ++ind)
	{
        	/* mem copy from host to device asynchcronously */
        	CUDA_SAFE_CALL( cudaMemcpyAsync(dAddMatMatA, hAddMatMatA, hA*wA*sizeof(double), cudaMemcpyHostToDevice,stream[ind]));
        	CUDA_SAFE_CALL( cudaMemcpyAsync(dAddMatMatB, hAddMatMatB, hB*wB*sizeof(double), cudaMemcpyHostToDevice, stream[ind]));
        	CUDA_SAFE_CALL( cudaMemcpyAsync(dAddMatMatC, hAddMatMatC, hA*wB*sizeof(double), cudaMemcpyHostToDevice, stream[ind]));
	}
	for( int ind=0; ind<nkernels; ++ind)
	{
                // queue nkernels  and record when they are done
                kernelMatMatAdd<<<dimGrid, dimBlock, 0, stream[ind]>>>(dAddMatMatA,dAddMatMatB, dAddMatMatC, hA,wA,BLOCK_SIZE);
        
	}

	for( int ind=0; ind<nkernels; ++ind)
	{
        	/* copy output from device to host */
        	CUDA_SAFE_CALL( cudaMemcpyAsync(hAddMatMatC, dAddMatMatC, hA*wB*sizeof(double), cudaMemcpyDeviceToHost, stream[ind]));
	}
       	CUDA_SAFE_CALL( cudaEventRecord(stopEvent));
        CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent));
       	CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
        
	/* get all errors from kernel launch */
        if ( err=cudaGetLastError())
        {
                printf(" File : %s , Line : %d , Error : %s \n",__FILE__, __LINE__, cudaGetErrorString(err));
        }


        /* calculate measured time and gflops */
	double tsecGpu;
        tsecGpu = (double) (elapsedTime  * 1.0e-3);          // converting to seconds from milliseconds

        /* check GPU results against CPU results */
        double errorNorm =  matMatAddCheckResult (hAddMatMatA,hAddMatMatB,hAddMatMatC,hA, wB);
        
	/* print output on screen */
        printf("%s\t%f\t %e\t\n","Asynchronous Concurrent Execution",tsecGpu,errorNorm);
	/* relese GPU events */

    	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);	

	
}


/************************************************************************
functions to execute multiple kernels without stream 
************************************************************************/
void funcSynchExec(double *dAddMatMatA, double *dAddMatMatB, double *dAddMatMatC,double *hAddMatMatA, double *hAddMatMatB, double *hAddMatMatC, int nkernels, int hA, int wA, int hB, int wB,  cudaDeviceProp deviceProp)
{
    	float elapsedTime;             // holds timing variables
    	cudaError_t     err;             // holds error value

    	/* create CUDA event handles */
	
	cudaEvent_t startEvent, stopEvent;
	CUDA_SAFE_CALL( cudaEventCreate(&startEvent));
        CUDA_SAFE_CALL( cudaEventCreate(&stopEvent));

        /* get all errors before  kernel launch */
        if ( err=cudaGetLastError())
        {
                printf(" File : %s , Line : %d , Error : %s \n",__FILE__, __LINE__, cudaGetErrorString(err));
        }

        /* define blocks and grids check grid and block dimension*/
        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE); //threads per block
        dim3 dimGrid(1,1); //blocks per grid
        checkBlockGridDim(deviceProp,dimGrid, dimBlock);

        /*Synchronous kernel execution */
        cudaEventRecord(startEvent, 0);
	for(int ind=0;ind<nkernels;ind++)
	{
        	/* mem copy from host to device asynchcronously */
        	CUDA_SAFE_CALL( cudaMemcpy(dAddMatMatA, hAddMatMatA, hA*wA*sizeof(double), cudaMemcpyHostToDevice));
        	CUDA_SAFE_CALL( cudaMemcpy(dAddMatMatB, hAddMatMatB, hB*wB*sizeof(double), cudaMemcpyHostToDevice));
        	CUDA_SAFE_CALL( cudaMemcpy(dAddMatMatC, hAddMatMatC, hA*wB*sizeof(double), cudaMemcpyHostToDevice));
	}
        for( int ind=0; ind<nkernels; ++ind)
        { 
                // queue nkernels  and record when they are done
                kernelMatMatAdd<<<dimGrid, dimBlock>>>(dAddMatMatA,dAddMatMatB, dAddMatMatC, hA,wA,BLOCK_SIZE);
        }
        for( int ind=0; ind<nkernels; ++ind)
	{
        	/* copy output from device to host */
	        CUDA_SAFE_CALL( cudaMemcpy(hAddMatMatC, dAddMatMatC, hA*wB*sizeof(double), cudaMemcpyDeviceToHost));
	}	
        /* in this sample we just wait until the GPU is done */
        CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
        CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );
        CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent) );

        /* get all errors from kernel launch */
        if ( err=cudaGetLastError())
        {
                printf(" File : %s , Line : %d , Error : %s \n",__FILE__, __LINE__, cudaGetErrorString(err));
        }


        /* calculate measured time and gflops */
        double tsecGpu = (double) (elapsedTime  * 1.0e-3);

        /* check CPU+GPU results against CPU results */
        double errorNorm =  matMatAddCheckResult (hAddMatMatA,hAddMatMatB,hAddMatMatC,hA, wB);
        
	/* print output on the screen */
	printf("%s\t\t\t%f\t %e\t\n","Synchronous Execution",tsecGpu,errorNorm);

	/* release GPU event */
    	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);	

}


/************************************************************
function to check the result with sequential result
***************************************************************/
double matMatAddCheckResult (double *hAddMatMatA,double *hAddMatMatB,double *outputGPU,int noOfRows, int noOfCols)
{

        int j, flag=0;               //Holds the relative error status
        double *outputCPU;            //Holds sequential resultant output
        double  errorNorm = 0.0;     // HOlds Error norm value
        double  eps=EPS;              
        double  relativeError=0.0;   // Holds relative error

        assert((outputCPU = (double *)malloc( sizeof(double) * noOfRows*noOfCols))!=NULL);

        /*sequential Matrix Matrix Addition result*/
        for( int ind=0 ; ind<noOfRows*noOfCols  ; ind++)
        {
                outputCPU[ind]= hAddMatMatA[ind] + hAddMatMatB[ind];
        }
        /* check opencl result with sequential result*/
        for(int ind=0 ; ind < noOfRows*noOfCols  ; ind++)
        {
                if (fabs(outputCPU[ind]) > fabs(outputGPU[ind]))
                        relativeError = fabs((outputCPU[ind] - outputGPU[ind]) / outputCPU[ind]);
                else
                        relativeError = fabs((outputGPU[ind] - outputCPU[ind]) / outputGPU[ind]);

                if (relativeError > eps)
                {
                        if(errorNorm < relativeError)
                        {
                                errorNorm = relativeError;
                                flag=1;
                        }
                }
        }
        if( flag == 1) {

                printf(" \n\t Results verfication : Failed");
                printf(" \n\t Considered machine precision : %e", eps);
                printf(" \n\t Relative Error                  : %e", errorNorm);

        }

        if(flag==0)
        {
        }
        free(outputCPU);
        return errorNorm;
}

void checkDeviceProperty(cudaDeviceProp deviceProp)
{	
	printf("\nDevice Used :\t %s",deviceProp.name);

        if( (deviceProp.concurrentKernels == 0 )) //check concurrent kernel support
        {
                printf("> GPU does not support concurrent kernel execution\n");
                printf("  CUDA kernel runs will be serialized\n");
        }
        if(deviceProp.asyncEngineCount == 0) //check concurrent data transfer support
        {
                printf("GPU does not support concurrent Data transer and overlaping of kernel execution & data transfer\n");
                printf("Mem copy call will be blocking calls\n");
        }
}

void checkDeviceAvailability()
{
	cudaError_t     err;             // holds error value
	err=cudaSetDevice(0);   //change this to set the code to another GPU
        if (err == cudaErrorDevicesUnavailable)
        {
               printf("\ndevice Not available\n");
               exit(0);
        }
}
void memoryAlloc(int hA, int hB, int wA, int wB )
{
        /* memory allocate to matrices*/
        CUDA_SAFE_CALL( cudaMallocHost((void**)&hAddMatMatA , hA * wA * sizeof(double)));
        CUDA_SAFE_CALL( cudaMallocHost((void**)&hAddMatMatB , hB * wB * sizeof(double)));
        CUDA_SAFE_CALL( cudaMallocHost((void**)&hAddMatMatC , hA * wB * sizeof(double)));

        /* initialize Matrices*/
        fillInData(hAddMatMatA,hA*wA);
        fillInData(hAddMatMatB,hB*wB);
        for(int index = 0; index < hA*wB ; index++)
                hAddMatMatC[index] = 0;


        /* allocate device memory*/
        CUDA_SAFE_CALL( cudaMalloc((void**) &dAddMatMatA, hA * wA *sizeof(double)));

        /* allocate device memory*/
        CUDA_SAFE_CALL( cudaMalloc((void**) &dAddMatMatB,hB * wB *sizeof(double)));

        /* allocate device memory*/
        CUDA_SAFE_CALL( cudaMalloc((void**) &dAddMatMatC,hA * wB *sizeof(double)));
}

void checkCmdlineArg(int argc,char* argv[])			
{
	switch(argc)
	{
		case 1:
			printf("\n Number of kernels not specified....default value will be taken as 16\n");
                        nkernels = 16;
			break;

		case 2 :
			nkernels = atoi(argv[1]);              // holds total number of concurrent kernels
			if(nkernels==0)
			{	
				printf("\nWrong input....\n");
				printf("\nUsage : <executable> [nkernels].........aborting \n");
				exit(-1);
			}

			if(nkernels > 16)
        		{
                		printf("\n The maximum number of kernel launches that a device can execute concurrently is 16 \n");
                		printf("\n Kernels will may not be executed concurrently...... \n");
        		} 

			break;
		default :
			 printf("\n Invalid options...\n");
			 printf("\n Usage : <./exe> [nKernels] \n");
			 exit(-1);
	}
}

/*****************************************************************************
                       main function
******************************************************************************/
int main(int argc, char *argv[])
{
	int cuda_device = 0;            // holds device number
	cudaStream_t *stream;           // holds stream array
        int  NSTREAM ;                  // holds total number of streams
        int hA, wA, hB, wB ,size;             //holds height and width for MatrixA and MatrixB
	hA = hB = HEIGHT;
	wA = wB = WIDTH;	
	int ind;
        ind =0;                       // holds inder variables
	checkCmdlineArg(argc,argv);				// function to check command line arguments

        NSTREAM = nkernels;
	
	/* function to check the device availability */
	checkDeviceAvailability(); 


	/* get device properties */
    	cudaDeviceProp deviceProp;
    	CUDA_SAFE_CALL( cudaGetDevice(&cuda_device));	
    	CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, cuda_device) );


	/* function to check device properties */
	checkDeviceProperty(deviceProp); // function to check device properties
	
	/* allocate and initialize an array of stream handles */
	for(ind = 0 ; ind < NSTREAM; ind++)
		stream = (cudaStream_t*) malloc(NSTREAM * sizeof(cudaStream_t));
        for(ind = 0; ind< NSTREAM; ind++)
               	CUDA_SAFE_CALL( cudaStreamCreate(&(stream[ind])));

      	/* function to allocate Host and Device matrices*/
	memoryAlloc(hA, hB, wA, wB);

	/* print information on the screen */	
	printf("\nNumber of kernels :\t %d", nkernels); 
	printf("\n\nNOTE : TIME_SEC includes data transfer time from host to device, device to host and kernel time");
        printf("\n\nExecution-Type\t\t\t\t Time_sec\t Relative-Error\n");
	printf("======================================================================\n");

	/* call function to execute Asynchronous kernels execution */
	funcAsynchConcurrentExec(dAddMatMatA, dAddMatMatB, dAddMatMatC,hAddMatMatA, hAddMatMatB, hAddMatMatC, nkernels, NSTREAM, stream , hA, wA, hB, wB,deviceProp);

	/* call function to execute  synchronous kernels execution */
	funcSynchExec(dAddMatMatA, dAddMatMatB, dAddMatMatC,hAddMatMatA, hAddMatMatB, hAddMatMatC, nkernels,hA, wA, hB, wB, deviceProp);

	/*********** Release all resources***************************/
	
	/* destroy an array of stream handles */
        for(ind = 0; ind< NSTREAM; ind++)
               	CUDA_SAFE_CALL( cudaStreamDestroy((stream[ind])));
	cudaFree(dAddMatMatA);
        cudaFree(dAddMatMatB);
        cudaFree(dAddMatMatC);
    	cudaDeviceReset(); // this will explicitly free all the resources
}
