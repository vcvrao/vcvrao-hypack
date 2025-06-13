
/*****************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example     :  cuda-matrix-vector-multiplication.cu.
 
  Objective   : Write CUDA program to compute Matrix-Vector multiplication using
		Synchronus execution and Asynchronus concurrent execution.

  Input       : Can specify the number of kernel. If not specified it will be 
                taken as 16 and more than 16 is not allowed.

  Output      : Execution time in seconds, Gflops achieved for both the 
                above mentioned execution.                                                                                                                           
  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

****************************************************************************/

#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<math.h>
#include<time.h>
#include<assert.h>

#define BLOCKSIZE 16
#define SIZE 1024
#define EPS 1.0e-15

cudaDeviceProp deviceProp;	

// Global declaration
double *host_Mat,*host_Vect,*host_ResVect,*cpu_ResVect;
double *device_Mat,*device_Vect,*device_ResVect;
int device_Count;
long int size = SIZE;
int matRowSize = size;
int vlength = size;
int matColSize = size;
int nkernels = 0;
int nstream;

/*mem error*/
void mem_error(char *arrayname, char *benchmark, int len, char *type)
{
        printf("\nMemory not sufficient to allocate for array %s\n\tBenchmark : %s  \n\tMemory requested = %d number of %s elements\n",arrayname, benchmark, len, type);
        exit(-1);
}

/*calculate Gflops*/
double calculate_gflops(float &Tsec)
{
      float gflops=(1.0e-9 * (( 2.0 * size*size )/Tsec));
		return gflops;
}

/*sequential function for mat vect multiplication*/
void CPU_MatVect()
{
	cpu_ResVect = (double *)malloc(matRowSize*sizeof(double));
	if(cpu_ResVect==NULL)
   	mem_error("cpu_ResVect","vectmatmul",size,"double");
	int i,j;
	for(i=0;i<matRowSize;i++)
	{
		cpu_ResVect[i]=0;
		for(j=0;j<matColSize;j++)
			cpu_ResVect[i]+=host_Mat[i*vlength+j]*host_Vect[j];
	}
}

/*Check for safe return of all calls to the device */
void CUDA_SAFE_CALL(cudaError_t call)
{
	cudaError_t ret = call;
   switch(ret)
   {
   	case cudaSuccess:
   		//printf("Success\n");                   
         break;
    /*case cudaErrorInvalidValue:                             
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
 

/*free memory*/
void dfree(double * arr[],int len)
{
	for(int i=0;i<len;i++)
   	CUDA_SAFE_CALL(cudaFree(arr[i]));
   printf("mem freed\n");
}

/* function to calculate relative error*/
void relError(double* dRes,double* hRes,int size)
{
	double relativeError=0.0,errorNorm=0.0;
   int flag=0;
   int i;
   for( i = 0; i < size; ++i) 
	{
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


/*prints the result in screen*/
void print_on_screen(char * program_name,float tsec,double gflops,int size,int flag)//flag=1 if gflops has been calculated else flag =0
{
	printf("\n---------------%s----------------\n",program_name);
   printf("\tSIZE\t TIME_SEC\t Gflops\n");
   if(flag==1)
   	printf("\t%d\t%f\t%lf\t",size,tsec,gflops);
   else
      printf("\t%d\t%lf\t%lf\t",size,"---","---");

}

/*funtion to check blocks per grid and threads per block*/
void check_block_grid_dim(cudaDeviceProp devProp,dim3 blockDim,dim3 gridDim)
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


/*Get the number of GPU devices present on the host */
int get_DeviceCount()
{
        int count;
        cudaGetDeviceCount(&count);
        return count;
}


/*Fill in the vector with double precision values */
void fill_dp_vector(double* vec,int size)
{
	int ind;
   for(ind=0;ind<size;ind++)
   	vec[ind]=drand48();
}


/////////////////////////////////////////////////////////////////////////////////////////
//
// MatVect : this kernel will perform actual MatrixVector Multiplication 
//
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void MatVectMultiplication(double *device_Mat, double *device_Vect,int matRowSize, int vlength,double *device_ResVect)
{
	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
   int tidy = blockIdx.y*blockDim.y + threadIdx.y;
   int tindex=tidx+gridDim.x*BLOCKSIZE*tidy;
   if(tindex<matRowSize)
	{
      int i;
		int m=tindex*vlength;
		device_ResVect[tindex]=0.00;
		for(i=0;i<vlength;i++)
			device_ResVect[tindex]+=device_Mat[m+i]*device_Vect[i];
	}
   __syncthreads();
}//end of MatVect device function



/*============================================================================================
		CheckCmdlineArg function for checking kernel size
===========================================================================================*/
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

/*==============================================================================================
						Check device Availability
=================================================================================================*/

void checkDeviceAvailability()
{
	cudaError_t err;             // holds error value
	err=cudaSetDevice(0);   //change this to set the code to another GPU
   if (err == cudaErrorDevicesUnavailable)
   {
      printf("\ndevice Not available\n");
      exit(0);
   }
}

/*=============================================================================================
				check device property for concurrent execution
=========================================================================================				*/
void checkDeviceProperty(cudaDeviceProp deviceProp)
{	
	printf("\n DEVICE USED :\t %s",deviceProp.name);
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
	
/*=============================================================================================
					Memory Allocation
===================================================================================					*/	
void memoryAlloc(long int size )
{
	/* memory allocate to matrices*/
   CUDA_SAFE_CALL( cudaMallocHost((void**)&host_Mat , matRowSize * matColSize * sizeof(double)));
   CUDA_SAFE_CALL( cudaMallocHost((void**)&host_Vect , vlength * sizeof(double)));
   CUDA_SAFE_CALL( cudaMallocHost((void**)&host_ResVect , vlength * sizeof(double)));
   /* initialize Matrices*/
   fill_dp_vector(host_Mat,matRowSize* matColSize);
   fill_dp_vector(host_Vect,vlength);
   for(int index = 0; index < matRowSize ; index++)
   	host_ResVect[index] = 0;
   /* allocate device memory*/
   CUDA_SAFE_CALL( cudaMalloc((void**) &device_Mat, matRowSize * matColSize *sizeof(double)));
   /* allocate device memory*/
   CUDA_SAFE_CALL( cudaMalloc((void**) &device_Vect,vlength*sizeof(double)));
   /* allocate device memory*/
   CUDA_SAFE_CALL( cudaMalloc((void**) &device_ResVect,matRowSize *sizeof(double)));
}

/***************************************************************
function to implement concurrent kernel execution 
***************************************************************/
void funcAsynchConcurrentExec(cudaStream_t *stream)
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
	
   /* Asynchronous kernel execution */
   int max=BLOCKSIZE*BLOCKSIZE;
	int BlocksPerGrid=matRowSize/max+1;
	dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
	if(matRowSize%max==0)
		BlocksPerGrid--;
	dim3 dimGrid(1,BlocksPerGrid);
	check_block_grid_dim(deviceProp,dimBlock,dimGrid);    

   //Starting the Asynchronus copy event
	cudaEventRecord(startEvent);
	
	for( int ind=0; ind<nkernels; ind++)
	{
	  	/* mem copy from host to device asynchcronously */
     	CUDA_SAFE_CALL( cudaMemcpyAsync(device_Mat, host_Mat, matRowSize*matColSize*sizeof(double), cudaMemcpyHostToDevice,stream[ind]));
      CUDA_SAFE_CALL( cudaMemcpyAsync(device_Vect, host_Vect, vlength*sizeof(double), cudaMemcpyHostToDevice, stream[ind]));
	}
	
	for( int ind=0; ind<nkernels; ind++)
	{
   	// queue nkernels  and record when they are done
      MatVectMultiplication<<<dimGrid, dimBlock, 0, stream[ind]>>>(device_Mat,device_Vect,matRowSize,vlength,device_ResVect);        
	}
	
	for( int ind=0; ind<nkernels; ind++)
	{
   	/* copy output from device to host */
      CUDA_SAFE_CALL( cudaMemcpyAsync(host_ResVect, device_ResVect, matRowSize*sizeof(double), cudaMemcpyDeviceToHost, stream[ind]));
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
	CPU_MatVect();
   relError(cpu_ResVect,host_ResVect,size);
	/* print output on screen */
   printf("%s\t%f\t \n"," Asynchronous Concurrent Execution",tsecGpu);
   printf("\n ----------------------------------------------------------------------\n");   

	/* relese GPU events */
   cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);	
}


/************************************************************************
functions to execute multiple kernels without stream 
************************************************************************/
void funcSynchExec()
{
	float elapsedTime;             // holds timing variables
   cudaError_t     err;             // holds error value
   /* create CUDA event handles */	
	cudaEvent_t startEvent, stopEvent;
	//create events
	CUDA_SAFE_CALL( cudaEventCreate(&startEvent));
   CUDA_SAFE_CALL( cudaEventCreate(&stopEvent));
   
	/* get all errors before  kernel launch */
   if ( err=cudaGetLastError())
   {
   	printf(" File : %s , Line : %d , Error : %s \n",__FILE__, __LINE__, cudaGetErrorString(err));
   }

   /* define blocks and grids check grid and block dimension*/
   int max=BLOCKSIZE*BLOCKSIZE;
	int BlocksPerGrid=matRowSize/max+1;
	dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
	if(matRowSize%max==0)
		BlocksPerGrid--;
	dim3 dimGrid(1,BlocksPerGrid);
	check_block_grid_dim(deviceProp,dimBlock,dimGrid);    
   
	/*Synchronous kernel execution */
   cudaEventRecord(startEvent, 0);
	
	for(int ind=0;ind<nkernels;ind++)
	{
	  	/* mem copy from host to device asynchcronously */
      CUDA_SAFE_CALL( cudaMemcpy(device_Mat, host_Mat, matRowSize*matColSize*sizeof(double), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL( cudaMemcpy(device_Vect, host_Vect, vlength*sizeof(double), cudaMemcpyHostToDevice));
	}
   
	for( int ind=0; ind<nkernels; ++ind)
   { 
   	// queue nkernels  and record when they are done
      MatVectMultiplication<<<dimGrid, dimBlock>>>(device_Mat,device_Vect,matRowSize,vlength, device_ResVect);
   }
   
	for( int ind=0; ind<nkernels; ++ind)
	{
   	/* copy output from device to host */
	   CUDA_SAFE_CALL( cudaMemcpy(host_ResVect, device_ResVect, matRowSize*sizeof(double), cudaMemcpyDeviceToHost));
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
   CPU_MatVect();
   relError(cpu_ResVect,host_ResVect,size);
	
	/* print output on the screen */
	printf("%s\t\t\t%f\t \t\n"," Synchronous Execution",tsecGpu);
   printf("\n ----------------------------------------------------------------------\n");    
	
	/* release GPU event */
  	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);	
}

/*==============================================================================================================
			main function by kp 2/5/2011
=====================================================================================================*/
int main(int argc, char *argv[])
{
		int ind = 0;
		int cuda_device = 0;            // holds device number
		cudaStream_t *stream;           // holds stream array
		
		checkCmdlineArg(argc,argv);				// function to check command line arguments
		
		/* function to check the device availability */
		checkDeviceAvailability();
 
		// Get device properties	
    	CUDA_SAFE_CALL( cudaGetDevice(&cuda_device));	
    	CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, cuda_device) );

		/* function to check device properties */
		checkDeviceProperty(deviceProp); // function to check device properties
		
		/* allocate and initialize an array of stream handles */
		nstream = nkernels;
		stream = (cudaStream_t*) malloc(nstream * sizeof(cudaStream_t));
      
		//creating streams
		for(ind = 0; ind < nstream; ind++)
               	CUDA_SAFE_CALL( cudaStreamCreate(&(stream[ind])));

		/* function to allocate Host and Device matrices*/
		memoryAlloc(size);

		/* print information on the screen */	
		printf("\n Number of kernels :\t %d", nkernels); 
		printf("\n\n NOTE : TIME_SEC includes data transfer time from host to device, device to host and kernel time");
    	printf("\n\n Execution-Type\t\t\t\t Time_sec\t Relative-Error\n");
		printf(" ======================================================================\n");

		/* call function to execute Asynchronous kernels execution */
		funcAsynchConcurrentExec(stream);

		/* call function to execute  synchronous kernels execution */
		funcSynchExec();

		/*********** Release all resources***************************/
		/* destroy an array of stream handles */
      for(ind = 0; ind< nstream; ind++)
               CUDA_SAFE_CALL( cudaStreamDestroy((stream[ind])));
		cudaFree(device_Mat);
      cudaFree(device_Vect);
      cudaFree(device_ResVect);
    	cudaDeviceReset(); // this will explicitly free all the resources
}
/*=================================================================================================*/
