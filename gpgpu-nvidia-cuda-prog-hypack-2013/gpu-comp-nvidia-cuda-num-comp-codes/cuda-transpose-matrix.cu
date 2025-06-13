/**********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example     : cuda-transpose-matrix.cu
 
  Objective   : Write CUDA program to perfomr Matrix-Transform operation.

  Input       : None 

  Output      : Execution time in seconds
                                                                                                                            
  Created    : August-2013

  E-mail     : hpcfte@cdac.in     
                                 
*********************************************************************/

#include<stdio.h>
#include<cuda.h>
#include<assert.h>

#define EPS 1.0e-12
#define GRIDSIZE 10
#define BLOCKSIZE 16

#define SIZE 128

int size = SIZE;
cudaDeviceProp deviceProp;	
cudaEvent_t start,stop;
cudaError_t ret;

/* kernel funtion */
__global__ void matTranspose(double *device_InMat, double *device_OutMat, int matRowColSize, int threadDim)
  {
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tindex = (threadDim * tidx) + tidy;
    int maxNumThread = threadDim * threadDim; 
    int pass = 0;  
    int rowCount ;
    int curColInd;

    while( (curColInd = (tindex + maxNumThread * pass))  < matRowColSize )
     {
        for( rowCount = 0; rowCount < matRowColSize; rowCount++)
          device_OutMat[curColInd * matRowColSize + rowCount] = device_InMat[rowCount* matRowColSize + curColInd];
        pass++;
      }
     __syncthreads();

  }
/*__global__ void add_matrix (double *matA,double *matB,double *matC,int length)
{
	int i=blockIdx.x * blockDim.x + threadIdx.x;
	int j=blockIdx.y * blockDim.y + threadIdx.y;
	int k = i+j*length;
	
	if(i<length&&j<length)
	matC[k] = matA[k]+matB[k];
	__syncthreads();
	
}*/

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



/* Get the number of GPU devices present on the host */
int get_DeviceCount()
{
	int count;
	cudaGetDeviceCount(&count);	
	return count;	
}

/* Fill in the vector with double precision values */
void fill_dp_matrix(double* vec,int size)
{
	int ind;
	for(ind=0;ind<size;ind++)
		vec[ind]=drand48();	
}


/* Function to check grid and block dimensions */
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

/* Function to print memory error */
void mem_error(char *arrayname, char *benchmark, int len, char *type)
{

	printf("\nMemory not sufficient to allocate for array %s\n\tBenchmark : %s  \n\tMemory requested = %d number of %s elements\n",arrayname, benchmark, len, type);
	printf("\tAborting\n");
	exit(-1);
}

/* launch kernel function is called in main() */
void launch_kernel_MatTranspose(double *device_InMat,double *device_OutMat,int size)
{

	dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
	dim3 dimGrid(size/dimBlock.x,size/dimBlock.y);	
 
        /* checking the maximum limit of blocksize and gridsize */
	check_block_grid_dim(deviceProp,dimBlock,dimGrid);
  
	matTranspose<<<dimGrid,dimBlock>>>(device_InMat,device_OutMat,size,BLOCKSIZE);
	
}

/* Function to calculate gflops */
double calculate_gflops(double &Tsec)
{
	//printf("time taken is %.8lf\n",Tsec);
	double gflops=(1.0e-9 * (( 1.0 * size*size )/Tsec));
	//printf("Gflops is \t%f\n",gflops);
	return gflops;

}

/* prints the result on screen */
void print_on_screen(char * program_name,float tsec,int size)
{
	printf("\n---------------%s----------------\n",program_name);
	printf("\tSIZE\t TIME_SEC\t\n");
	printf("\t%d\t%f\t",size,tsec);

}

/* Function to perform Mat Addition on CPU */
/*void CPU_MatMatAdd(double *A,double *B,double *C,int length)
{
	for(int i =0;i<length*length;i++)
		C[i] = A[i]+B[i];
}*/

/* Function to check cpu and gpu results */
/* void relError(double* dRes,double* hRes,int size)
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

}*/

/* free memory */
void dfree(double * arr[],int len)
{
        for(int i=0;i<len;i++)
                CUDA_SAFE_CALL(cudaFree(arr[i]));
}

/************************************************************
function to check the result with sequential result
***************************************************************/
int matTransposeCheckResult(double *host_InMat,double *host_OutMat,int rows,int cols)
{
        int i,count,flag=0;
        double *temp_out;

        double  eps=EPS;
        double  relativeError=0.0;
        double  errorNorm = 0.0;

        assert((temp_out = (double *)malloc( sizeof(double) * rows*cols))!=NULL);
        int colIndex=0;
        while(colIndex != rows)
        {
                for(count = 0 ; count < cols; count++ )
                {
                        temp_out[colIndex+rows*count] =  host_InMat[count + rows   * colIndex];
                }
                colIndex++;
        }

        /*** check relative error with approx precision ****/
        for( i = 0; i < rows*cols; ++i)
        {

                if (fabs(temp_out[i]) > fabs(host_OutMat[i]))
                        relativeError = fabs((temp_out[i] - host_OutMat[i]) / temp_out[i]);
                else
                        relativeError = fabs((host_OutMat[i] - temp_out[i]) / host_OutMat[i]);

                if (relativeError > eps && relativeError != 0.0e+00 )
                {
                        if(errorNorm < relativeError)
                        {
                                errorNorm = relativeError;
                                flag=1;
			  }
                }

        }

        if( flag == 1) {

                printf(" \n \t\t Results verfication : Failed\n");
                printf(" \n \t\t Considered machine precision : %e\n", eps);
                printf(" \n \t\t Relative Error                  : %e\n", errorNorm);

        }
        else
        {

                printf("\n \n\t\tResults verfication : Success\n\n");
        }

        free(temp_out);
	return 0;
}




/* main()   */

int main()
{
	double *host_InMat,*host_OutMat;
	double *device_InMat,*device_OutMat;

	int device_Count=get_DeviceCount();
        printf("\n\nNUmber of Devices : %d\n\n", device_Count);

        /* Device Selection, Device 1 */
        cudaSetDevice(0);
	
	int device;
        /* Current Device Detection */
        cudaGetDevice(&device);         
        cudaGetDeviceProperties(&deviceProp,device);

	printf("Using device %d: %s \n", device, deviceProp.name);

	/* event creation */
	CUDA_SAFE_CALL(cudaEventCreate (&start));
        CUDA_SAFE_CALL(cudaEventCreate (&stop));
   

   
       /* allocating the memory for each matrix */
	host_InMat = new double[size*size];
	host_OutMat = new double[size*size];
	 if(host_InMat==NULL)
                mem_error("host_InMat","mattranspose",size,"double");

	if(host_OutMat==NULL)
                mem_error("host_OutMat","mattranspose",size,"double");


	/* filling the matrix with double precisio */
  	fill_dp_matrix(host_InMat,size*size);

	/* filling host_MatC with 0.0 value */
	for(int i =0;i<size*size ;i++)
	host_OutMat[i]=0.0;
  
 	/* allocating memory on GPU */
	CUDA_SAFE_CALL(cudaMalloc( (void**)&device_InMat,size*size*sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc( (void**)&device_OutMat, size*size*sizeof(double)));
 	
	/* copying host matrix to device matrix */
    	CUDA_SAFE_CALL(cudaMemcpy((void*)device_InMat, (void*)host_InMat, size*size* sizeof(double) , cudaMemcpyHostToDevice ));
    	CUDA_SAFE_CALL(cudaMemcpy((void*)device_OutMat, (void*)host_OutMat, size*size*sizeof(double) , cudaMemcpyHostToDevice ));
  

	CUDA_SAFE_CALL(cudaEventRecord (start, 0));
	launch_kernel_MatTranspose(device_InMat,device_OutMat,size);               //launching the kernel
	CUDA_SAFE_CALL(cudaEventRecord (stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize (stop));

	

	/* computing elapsed time */
	float elapsedTime;	
	CUDA_SAFE_CALL(cudaEventElapsedTime ( &elapsedTime, start, stop));
	double Tsec = elapsedTime *1.0e-3;


	/* calling funtion for measuring Gflops */
	//calculate_gflops(Tsec);
	
        /* printing the result on screen */
        print_on_screen("MAT TRANSPOSE",Tsec,size);
       
   	/* retriving result from device */
      CUDA_SAFE_CALL(cudaMemcpy((void*)host_OutMat, (void*)device_OutMat, size*size*sizeof(double) , cudaMemcpyDeviceToHost ));


  	/* to get the result uncomment this part
   printf("\n ----------------------------------------------------------------------");	
	for(int i =0;i<size*size;i++)
	   printf("%lf", host_OutMat[i]);*/


	/* comparing result of CPU-GPU */
	matTransposeCheckResult(host_InMat,host_OutMat,size,size);


   
	/* free the device memory */
	double *array[2];
	array[0]=device_InMat;
	array[1]=device_OutMat;
	//array[2]=device_MatC;
	
	dfree(array,2);
	
	/* free host memory */

	   free(host_InMat);
	   free(host_OutMat);
	   //free(host_MatC);
	  // free(CPU_Result);

	   cudaDeviceReset();

 }// end of main
