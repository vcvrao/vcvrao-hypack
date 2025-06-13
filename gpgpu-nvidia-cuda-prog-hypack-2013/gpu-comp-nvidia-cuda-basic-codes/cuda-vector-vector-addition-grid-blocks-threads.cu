
/***********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example     :  cuda-vector-vector-addition-grid-blocks-threads.cu
 
  Objective   : Write a CUDA  program to compute Vector-Vector Addition 
                using global memory implementation.                 

  Input       : None 

  Output      : Execution time in seconds , Gflops achieved

  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

************************************************************************/

#include<stdio.h>
#include<cuda.h>
#include<math.h>

#define EPS 1.0e-12
#define BLOCKSIZE 16
#define SIZE 128

/* global variable declaration */
cudaDeviceProp deviceProp;  // holds device property

/* kernel to calculate vector vector addition */
__global__ void vectvectadd(double* dm1,double* dm2,double *dres,int num)
{
        int tx = blockIdx.x * blockDim.x + threadIdx.x;
        int ty = blockIdx.y * blockDim.y + threadIdx.y;
        int tindex=tx + (gridDim.x) * (blockDim.x) * ty;

        if(tindex < num)
	dres[tindex] = dm1[tindex] + dm2[tindex];
}

/* Check for safe return of all calls to the device */
void CUDA_SAFE_CALL(cudaError_t call)
{
        cudaError_t ret = call;
        switch(ret)
        {
                case cudaSuccess:
                //              printf("Success\n");
                                break;
                default:
                        {
                                printf(" ERROR at line :%i.%d' ' %s\n",__LINE__,ret,cudaGetErrorString(ret));
                                exit(-1);
                                break;
                        }
        }
}


/* Fill in the vector with double precision values */
void fill_dp_vector(double* vec,int size)
{
        int index;
        for(index=0 ; index < size ; index++)
                vec[index]=drand48();
}


/* checl grid and block dimensions */
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

/* function to check memory error */
void mem_error(char *arrayname, char *benchmark, int len, char *type)
{

        printf("\nMemory not sufficient to allocate for array %s\n\tBenchmark : %s  \n\tMemory requested = %d number of %s elements\n",arrayname, benchmark, len, type);
        printf("\n\tAborting\n\n");
        exit(-1);
}

/* Get the number of GPU devices present on the host */
int get_DeviceCount()
{
        int count;
        cudaGetDeviceCount(&count);
        return count;
}

/* function to calculate relative error */
void relError(double* dRes,double* hRes,int size)
{
        double relativeError=0.0,errorNorm=0.0;
        int flag=0;
        int index;

        for( index = 0; index < size; ++index) {
                if (fabs(hRes[index]) > fabs(dRes[index]))
                        relativeError = fabs((hRes[index] - dRes[index]) / hRes[index]);
                else
                        relativeError = fabs((dRes[index] - hRes[index]) / dRes[index]);

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
                printf(" \n\t Results verfication : Failed");
                printf(" \n \tConsidered machine precision : %e", EPS);
                printf(" \n \tRelative Error                  : %e\n", errorNorm);

        }
        else
                printf("\n \tResults verfication : Success\n");

}



/* function to check device availability */
void deviceQuery()
{
	 int device_Count;
        device_Count = get_DeviceCount();
        printf("\n\nNUmber of Devices : %d\n\n", device_Count);

        cudaSetDevice(0);
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&deviceProp,device);
        printf("Using device %d: %s \n", device, deviceProp.name);
}

/* Launch kernel */
void launch_kernel(double *dVectA , double *dVectB , double *dresult , int vlength)
{
	 dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);              // holds number of blocks per grid
         dim3 dimGrid((vlength/BLOCKSIZE*BLOCKSIZE)+1,1);  // holds number of threads per block
	
	/* check block and grid dimension */ 
         check_block_grid_dim(deviceProp,dimBlock,dimGrid);
       
	/* Launch vector vector additon kernel */
	 vectvectadd<<<dimGrid, dimBlock>>>(dVectA, dVectB, dresult,vlength );
}

/* Function to print gflosp rating */
double print_Gflops_rating(float Tsec,int size)
{
	double gflops;	
        gflops=(1.0e-9 * (( 1.0 * size )/Tsec));
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


/* Function to perform computation on CPU*/
void vectVectAdd_Seq(double *A ,double *B,double *C,int size)
{
	for(int index=0 ; index < size ; index++)
	C[index] = A[index] + B[index];
}


/* main function */
int main()
{
	double *dVectA, *dVectB; // holds device input vector
	double *hVectA, *hVectB; // holds host input vector
	double *dresult;         // holds device resultant vector
	double  *hVectC;         // holds host resultant vector  
	double *cpuVectC;        //holds host-cpu resultant vector 
	int   vlength ;          // holds vector length
	cudaEvent_t start,stop;  // holds cuda event
	vlength = SIZE;          //holds vector length

	/* check device availability */
	deviceQuery();

	/* create cuda events */
	CUDA_SAFE_CALL(cudaEventCreate (&start));
        CUDA_SAFE_CALL(cudaEventCreate (&stop));
     
 	/* allocation pageable host memory */
   	hVectA = (double*) malloc( vlength *  sizeof(double));
   	hVectB = (double*) malloc( vlength * sizeof(double));
   	hVectC = (double*) malloc( vlength * sizeof(double));
   	cpuVectC = (double*) malloc( vlength * sizeof(double));

	/* check for memory error */
	 if(hVectA==NULL)
                mem_error("hVectA","vectvectmul",vlength,"double");

	 if(hVectB==NULL)
                mem_error("hVectB","vectvectmul",vlength,"double");

	 if(hVectC==NULL)
                mem_error("hVectC","vectvectmul",vlength,"double");

   	/* allocation device memory */
   	CUDA_SAFE_CALL(cudaMalloc( (void**)&dVectA, vlength * sizeof(double)));
   	CUDA_SAFE_CALL(cudaMalloc( (void**)&dVectB, vlength * sizeof(double)));
   	CUDA_SAFE_CALL(cudaMalloc( (void**)&dresult, vlength*sizeof(double)));
  
	/* fill host vectors with random generated double precision value */
       	fill_dp_vector(hVectA,vlength);
	fill_dp_vector(hVectB,vlength); 
	for(int index=0 ; index < vlength ; index++)
	{
		hVectC[index] = 0.0;
	}
	for(int index=0 ; index < vlength ; index++)
	{
		cpuVectC[index]=0.0;
	}
	  
	/* copy host vector to device vector */
    	CUDA_SAFE_CALL(cudaMemcpy((void*)dVectA, (void*)hVectA, vlength* sizeof(double) , cudaMemcpyHostToDevice ));
    	CUDA_SAFE_CALL(cudaMemcpy((void*)dVectB, (void*)hVectB, vlength* sizeof(double) , cudaMemcpyHostToDevice ));
 

   	/* start time */
	CUDA_SAFE_CALL(cudaEventRecord (start, 0));

	/* function to launch vector vector addition kernel */
	launch_kernel(dVectA,dVectB,dresult,vlength); 

	/* time ends */
   	CUDA_SAFE_CALL(cudaEventRecord (stop, 0));
   	CUDA_SAFE_CALL(cudaEventSynchronize (stop));

	/* copy resultant vector from device to host */
	CUDA_SAFE_CALL(cudaMemcpy((void*)hVectC, (void*)dresult,vlength*sizeof(double) , cudaMemcpyDeviceToHost ));

	/* computing elapsed time  */
	float elapsedTime;
	double Tsec;      
        CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));

        Tsec = elapsedTime *1.0e-3; //time in sec now

	/* calculation of Gflops */
        print_Gflops_rating(Tsec,vlength);

	/* printing the result on screen */
        print_on_screen("vect vect ADDITION",Tsec, print_Gflops_rating(Tsec,vlength),vlength,1);
	
	/* uncomment to get the sequential result of vector vector additoion */
	vectVectAdd_Seq(hVectA ,hVectB,cpuVectC,vlength);


 	/* uncomment to compare host-cpu results with gpu results */
	 relError(hVectC,cpuVectC,vlength);


	/* free host memory */
   	free(hVectA);
   	free(hVectB);
   	free(hVectC);
   	free(cpuVectC);
	/* free device memory */
	cudaFree(dVectA);	
	cudaFree(dVectB);	
	cudaFree(dresult);
}
