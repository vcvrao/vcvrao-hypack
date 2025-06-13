
/**********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example     : cuda-infinity-norm.cu
 
  Objective   : Write CUDA program to calculate Infinity Norm of Matrix.

  Input       : None 

  Output      : Data Size ,Execution time in seconds
                                                                                                                            
  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

**********************************************************************/

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
__global__ void matInfinityNorm(double *device_InMat,double *device_InfinityNorm,int matRowSize, int matColSize, int threadDim)
  {
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tindex = (threadDim * tidx) + tidy;
    int maxNumThread = threadDim * threadDim; 
    int pass = 0;  
    int colCount, tCount ;
    int curRowInd;
    double tempInfinityNorm = 0.0;
    double rowMaxValue = 0.0;
      
    for( tCount = 1; tCount < maxNumThread; tCount++)
         device_InfinityNorm[tCount] = 0.0; 

    while( (curRowInd = (tindex + maxNumThread * pass))  < matRowSize )
     {
        rowMaxValue = 0.0;
        for( colCount = 0; colCount < matColSize; colCount++)
          rowMaxValue += fabs(device_InMat[curRowInd* matRowSize + colCount]);
        tempInfinityNorm = ( tempInfinityNorm>rowMaxValue? tempInfinityNorm:rowMaxValue);
        pass++;
      }

    device_InfinityNorm[ tindex ] = tempInfinityNorm;
     __syncthreads();
   
    if(tindex == 0) 
      for( tCount = 1; tCount < maxNumThread; tCount++)
         device_InfinityNorm[0] = device_InfinityNorm[0]> device_InfinityNorm[tCount]? device_InfinityNorm[0]: device_InfinityNorm[tCount]; 


}



/* Check for safe return of all calls to the device */
void CUDA_SAFE_CALL(cudaError_t call)
{
        cudaError_t ret = call;
        switch(ret)
        {
                case cudaSuccess:
                                break;
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
void fill_dp_matrix(double* mat,int size)
{
	int ind;
	for(ind=0;ind<size;ind++)
		mat[ind]=drand48();	
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
void launch_kernel_MatInfinityNorm(double *device_InMat,double *device_InfinityNorm,int size)
{

	dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
	dim3 dimGrid(size/dimBlock.x,size/dimBlock.y);	
 
        /* checking the maximum limit of blocksize and gridsize */
	check_block_grid_dim(deviceProp,dimBlock,dimGrid);
  
	matInfinityNorm<<<dimGrid,dimBlock>>>(device_InMat,device_InfinityNorm,size,size,BLOCKSIZE);
	
}

/* Function to calculate gflops */
double calculate_gflops(double &Tsec)
{
	double gflops=(1.0e-9 * (( 1.0 * size*size )/Tsec));
	return gflops;

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


/* free memory */
void dfree(double * arr[],int len)
{
        for(int i=0;i<len;i++)
                CUDA_SAFE_CALL(cudaFree(arr[i]));
}

/************************************************************
function to check the result with sequential result
***************************************************************/

int matInfinityNormCheckResult(double *host_InMat,int rows,int cols,double *host_InfinityNorm)
{
	int currRow,iCol,flag=0;
	double temp_norm =0.0,myRowSum=0.0;

	double  eps=EPS;
	double  relativeError=0.0;
	double  errorNorm = 0.0;

	/* sequential code to calculate Infinity norm*/
	for (currRow=0;currRow<rows;currRow++)
       	{
         
            myRowSum = 0;
            for(iCol = 0 ;iCol < cols; iCol++)
            myRowSum += host_InMat[currRow*cols+iCol];

            if(temp_norm < myRowSum )
            temp_norm = myRowSum;
       }


//	 check opencl result with sequential code 
	if (fabs(temp_norm) > fabs(*host_InfinityNorm))
                        relativeError = fabs((temp_norm - *host_InfinityNorm) / temp_norm);
                else
                        relativeError = fabs((*host_InfinityNorm - temp_norm) / *host_InfinityNorm);

                if (relativeError > eps && relativeError != 0.0e+00 )
                {
                        if(errorNorm < relativeError)
                        {
                                errorNorm = relativeError;
                                flag=1;
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
	return 0;
}


/* main()   */

int main()
{
	double *host_InMat,*host_InfinityNorm;
	double *device_InMat,*device_InfinityNorm;

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
	host_InfinityNorm = new double[sizeof(double)];
	 if(host_InMat==NULL)
                mem_error("host_InMat","matInfinityNorm",size,"double");

	if(host_InfinityNorm==NULL)
                mem_error("host_InfinityNorm","matInfinityNorm",sizeof(double),"double");


	/* filling the matrix with double precisio */
  	fill_dp_matrix(host_InMat,size*size);
  
 	/* allocating memory on GPU */
	CUDA_SAFE_CALL(cudaMalloc( (void**)&device_InMat,size*size*sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc( (void**)&device_InfinityNorm,sizeof(double)));
 	
	/* copying host matrix to device matrix */
    	CUDA_SAFE_CALL(cudaMemcpy((void*)device_InMat, (void*)host_InMat, size*size* sizeof(double) , cudaMemcpyHostToDevice ));
    	CUDA_SAFE_CALL(cudaMemcpy((void*)device_InfinityNorm, (void*)host_InfinityNorm, sizeof(double) , cudaMemcpyHostToDevice ));
  

	CUDA_SAFE_CALL(cudaEventRecord (start, 0));
	launch_kernel_MatInfinityNorm(device_InMat,device_InfinityNorm,size);               //launching the kernel
	CUDA_SAFE_CALL(cudaEventRecord (stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize (stop));

	

	/* computing elapsed time */
	float elapsedTime;	
	CUDA_SAFE_CALL(cudaEventElapsedTime ( &elapsedTime, start, stop));
	double Tsec = elapsedTime *1.0e-3;


	/* calling funtion for measuring Gflops */
	 calculate_gflops(Tsec);
	
        /* printing the result on screen */
        print_on_screen("MAT INFINTYNORM",Tsec,calculate_gflops(Tsec),size,1);
       
   	/* retriving result from device */
        CUDA_SAFE_CALL(cudaMemcpy((void*)host_InfinityNorm, (void*)device_InfinityNorm, sizeof(double) , cudaMemcpyDeviceToHost ));


  	/* to get the result uncomment this part
   	printf("\n ----------------------------------------------------------------------");	
	printf("InfinityNorm = %lf", *host_InfinityNorm);*/


	/* comparing result of CPU-GPU */
	matInfinityNormCheckResult(host_InMat,size,size,host_InfinityNorm);
   
	/* free the device memory */
	double *array[2];
	array[0]=device_InMat;
	array[1]=device_InfinityNorm;
	
	dfree(array,2);
	
	/* free host memory */

	free(host_InMat);
	free(host_InfinityNorm);

	cudaDeviceReset();

 }// end of main
