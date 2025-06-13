/**
 *This file executes compute intensive kernel 
**/

#include"cuda_nvml_power_kernel_functions.h"

/*kernel funtion*/
/**
 * kernel to execute Matrix Matrix multiplication 
 * @param device MatrixA
 * @param device MatrixB
 * @param device MatrixC
 * @param size of matrix A
 * @param size of Matrix B
 * @param Result Matrix 
**/
__global__ void Muld(double* matA, double* matB, int wA, int wB, double* matC)
  {
   int blockIDx = blockIdx.x;
   int blockIDy = blockIdx.y;
   int threadIDx = threadIdx.x;
   int threadIDy = threadIdx.y;
   int aBegin  = wA * BLOCKSIZE * blockIDy;
   int aEnd = aBegin + wA - 1;
   int aStep  = BLOCKSIZE;
   int bBegin  =  BLOCKSIZE * blockIDx;
   int bStep = BLOCKSIZE * wB;
   double tmpData = 0;
   for(int a = aBegin, b = bBegin; a <= aEnd ; a  += aStep, b += bStep)
     {
       __shared__ double As[BLOCKSIZE][BLOCKSIZE];
       __shared__ double Bs[BLOCKSIZE][BLOCKSIZE];
       As[threadIDy][threadIDx] = matA[a + wA * threadIDy + threadIDx];
       Bs[threadIDy][threadIDx] = matB[b+ wB * threadIDy + threadIDx];
       __syncthreads();
       for(int k= 0; k< BLOCKSIZE; ++k)
         tmpData += As[threadIDy][k] * Bs[k][threadIDx];
       __syncthreads();
      }
     int c = wB * BLOCKSIZE * blockIDy + BLOCKSIZE * blockIDx;
     matC[ c+ wB * threadIDy + threadIDx] = tmpData;
                                
  }/* end of Muld device code */



/**
 * This function compares device computed result with host computed result
 * and find out relative error between both. 
 * @param host matrix A
 * @param host matrix B
 * @param result : Bbtained from device computation
 * @param Size of matrix A
 * @param Size of matrix B
**/
 
int matMatMultCheckResultGMDP (double *hMatA, double *hMatB,double *output, int rows, int cols)
{

        int i,j,k,step=0;
        double *tmpOut;
        double sum;
        double  errNorm = 0.0;
        double  eps=EPS;
        double  relativeError=0.0;
        int     flag=0;

        assert((tmpOut = (double *)malloc( sizeof(double) * rows * rows))!=NULL);
        /*calculate sequential result*/
        for( i=0 ; i<rows ; i++)
        {
                for( j=0 ; j<rows  ; j++)
                {
                        sum = 0.00;
                        for( k=0 ; k<cols  ; k++)
                        {
                                sum += hMatA[i * cols   + k] * hMatB[k * rows   + j];
                        }
                        tmpOut[step++] = sum;
                }
        }
        /* check opencl result with sequential result*/
        for( i=0 ; i < rows  ; i++)
        {
                for( j=0 ; j < rows  ; j++)
                {
                        if (fabs(tmpOut[i*rows +j]) > fabs(output[i*rows +j]))
                        relativeError = fabs((tmpOut[i*rows +j] - output[i*rows +j]) / tmpOut[i*rows +j]);
                        else
                        relativeError = fabs((output[i*rows +j] - tmpOut[i*rows +j]) / output[i*rows +j]);

                        if (relativeError > eps && relativeError != 0.0e+00 )
                        {
                                if(errNorm < relativeError)
                                {
                                        errNorm = relativeError;
                                        flag=1;
                                }
                        }
                 }
         }
        if( flag == 1) {

                printf(" \n Results verfication : Failed");
                printf(" \n Considered machine precision : %e", eps);
                printf(" \n Relative Error                  : %e", errNorm);

        }
        else
        {

                printf("\n\t\t\t Results verfication : Success\n\n");
        }

        free(tmpOut);
        return 0;
}


/**  
 * function to free device memory
 * @param array : whose memory needs to free
 * @param size of array 
**/
void dfree(double * arr[],int len)
{
        for(int i=0;i<len;i++)
                CUDA_SAFE_CALL(cudaFree(arr[i]));
}


/**
 * function used to check memory errors during memory allocation time 
 * @param array: whose memory is checked 
 * @param function name: in which memory error occured
 * @param size of array
 * @param 
**/
void memErr(char *arrayname, char *benchmark, int len, char *numElements)
{                               
        printf("\nMemory not sufficient to allocate for array %s\n\tBenchmark : %s  \n\tMemory requested = %d number of %s elements\n",arrayname, benchmark, len, numElements);
        exit(-1);       
}       



/** 
 * fucntion to check GRID & BLOCK dimension. 
 * Check whether these number of blocks per grid
 * or threads per block can be assigned. 
 * @param device property: device property of available cuda device
 * @param block dimension: what ever is mentioned 
 * @param grid dimension: what ever is mentioned 
**/
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


/**
 * This function calculates Gflops of compute intensive kernel
 * @param Execution time: total time for execution of kernel
 * @return gflops
**/
double calGFlops(double &Tsec)
{
        //printf("time taken is %.8lf\n",Tsec);
        double gFlops=(1.0e-9 * (( 1.0 * size*size*size )/Tsec));
        //printf("Gflops is \t%f\n",gFlops);
        return gFlops;

}


/**
 * Function to print output on consol
**/
void printOnScreen(char * program_name,float tsec,double gFlops,int size,int flag)//flag=1 if gFlops has been calculated else flag =0
{
        printf("\n---------------%s----------------\n",program_name);
        printf("\tSIZE\t TIME_SEC\t Gflops\n");
        if(flag==1)
        printf("\t%d\t%f\t%lf\t\n",size,tsec,gFlops);
        else
        printf("\t%d\t%lf\t%lf\t\n",size,"---","---");

}


/**
 * launch kernel means define number of blocks per grid,
 * threads per block and then call kernel with these 
 * configurations and calculate time & gflops for execution of kernel.
**/
void kernelMatMult()
{
/* threads_per_block= BLOCKSIZE, blocks_per_grid=size/dimBlock  */

        dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
        dim3 dimGrid(size/dimBlock.x,size/dimBlock.y);

	//checking the maximum limit of blocksize and gridsize-------------------
        checkBlockGridDim(deviceProp,dimBlock,dimGrid);
        
	cudaEventRecord(start,0);
        Muld<<<dimGrid,dimBlock>>>(dMatA,dMatB,size,size,dMatC);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime,start,stop);
        
	Tsec=elapsedTime*1.0e-3;
        calGFlops(Tsec);
	//printOnScreen("MAT MAT Mult",Tsec,calGFlops(Tsec),size,1);
}

/**
 * fill matrix (2-d vector) with random values
 * @param host vector
 * @param size of matrix
**/ 
void fillVectorDP(double* vec,int size)
{
        int ind;
        for(ind=0;ind<size;ind++)
                vec[ind]=drand48();
}



/**
 * Allocate memory on host & device. 
 * transfer data from host to device then call
 * kernel launch function and transfer result 
 * matrix from device to host. Then free all 
 * memories and reset device.
 * @param thread id 
**/
void *mat_mult(void *t)
{
		
	/* kernel launching section starts */
	/*******************************************************************************************/
	int device;
	cudaDeviceReset();
	cudaSetDevice(0);
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&deviceProp,device);

        //event creation...
        CUDA_SAFE_CALL(cudaEventCreate (&start));
        CUDA_SAFE_CALL(cudaEventCreate (&stop));
	
        hMatA = (double *)malloc(sizeof(double) * size*size);
        hMatB = (double *)malloc(sizeof(double) * size*size);
        hMatC = (double *)malloc(sizeof(double) * size*size);
         if(hMatA==NULL)
                memErr("hMatA","MatMatMult",size,"double");

        if(hMatB==NULL)
                memErr("hMatB","MatMatMult",size,"double");
        if(hMatC==NULL)
                memErr("hMatC","MatMatMult",size,"double");


        //--------filling the matrix with double precision-----------
        fillVectorDP(hMatA,size*size);
        fillVectorDP(hMatB,size*size);

        //allocating memory on GPU
        CUDA_SAFE_CALL(cudaMalloc( (void**)&dMatA,size*size*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc( (void**)&dMatB, size*size*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc( (void**)&dMatC,size*size*sizeof(double)));

        // copying host matrix to device matrix
        CUDA_SAFE_CALL(cudaMemcpy((void*)dMatA, (void*)hMatA, size*size* sizeof(double) , cudaMemcpyHostToDevice ));
        CUDA_SAFE_CALL(cudaMemcpy((void*)dMatB, (void*)hMatB, size*size*sizeof(double) , cudaMemcpyHostToDevice ));

        kernelMatMult(); //launching the kernel


        //retriving result from device
        CUDA_SAFE_CALL(cudaMemcpy((void*)hMatC, (void*)dMatC, size*size*sizeof(double) , cudaMemcpyDeviceToHost ));


	cudaThreadSynchronize();

	/* kernel launching section ends */
	/*******************************************************************************************/

	//matMatMultCheckResultGMDP(hMatA,hMatB,hMatC,size,size);	
	//free the device memory----------
        double *array[3];
        array[0]=dMatA;
        array[1]=dMatB;
        array[2]=dMatC;

        dfree(array,3);
	dMatA = NULL;
	dMatB = NULL;
	dMatC = NULL;
        //free host memory---------- 
        
        free(hMatA);	
	hMatA = NULL;	
        free(hMatB);
	hMatB = NULL;	
        free(hMatC);
	hMatC = NULL;	
	cudaDeviceReset();
	sleep(25);
	sigFlag = 0;   //conditional variable
	sleep(10);
	printOnScreen("MAT MAT Mult",Tsec,calGFlops(Tsec),size,1);
	pthread_exit(NULL);
}

