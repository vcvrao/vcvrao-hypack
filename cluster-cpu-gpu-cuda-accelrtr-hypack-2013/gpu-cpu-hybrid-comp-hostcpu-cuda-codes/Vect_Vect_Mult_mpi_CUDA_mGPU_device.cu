/****************************************************************************

                C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Objective   : program to solve a Vector Vector multiplication using block
                striped partioning on hybrid computing using MPI C++ and CUDA
                and Multiple GPUs 

  Input       : Process 0 initialize the Vector.

  Output      : Process 0 prints the resultant value.

  Necessary      Size of the  Each Vector should be greater than number of
  Conditons   :  processes and perfectly divisible by number of processes.

  Created     : August-2013

  E-mail      : hpcfte@cdac.in

****************************************************************************/

#include<stdio.h>
#include<cuda.h>
#include<error.h>
#define BLOCKSIZE 16
#define SIZE 1024

int vlen=SIZE;
int size=SIZE;

float *hVectA,*hVectB,hRes;
float elapsedTime,elapsedTime1;

double Tsec,Tsec1,gflops,gflops1;
cudaEvent_t start,stop,start1,stop1;

int blocksPerGrid;
int gridsPerBlock;

void routine(void * givendata);
void init(int s);

struct Data
{
        int deviceId;
        int size;
        float* a;
        float* b;
        float retVal;
};
Data vector[2];


/*sequential function*/
extern "C" float compare()
{
	init(vlen);
	float sum=0;
	for(int i=0;i<vlen;i++)
	{
		sum+=hVectA[i]*hVectB[i];
	}
	printf("cpu_sum=%f\n",sum);
	return sum;
}

/* kernel to execute vector vector multiplication */
__global__ void vvmul(int len,float* A,float* B,float *C)
{
        int tid= blockIdx.x*blockDim.x*blockDim.y + threadIdx.x +threadIdx.y * blockDim.x;

        while(tid < len)
        {
                C[tid] = A[tid] * B[tid];
                tid += blockDim.x * gridDim.x;
        }
}

/* function display() */
void display(float* arr,int size)
{
        int i;
        for(i=0;i<size;i++)
                printf("%f ",arr[i]);
        printf("\t%d\n",i);
}
/*
extern "C"
void result()
{
	printf("Kernel execution done\n");

        hRes=vector[0].retVal + vector[1].retVal;
        printf("The product of two vectors is :%f\n",hRes);

        free(hVectA);
        free(hVectB);

}
*/


/*mem error*/
void mem_error(char *arrayname, char *benchmark, int len, char *type)
{
        printf("\nMemory not sufficient to allocate for array %s\n\tBenchmark : %s  \n\tMemory requested = %d number of %s elements\n",arrayname, benchmark, len, type);
        exit(-1);
}


/*cuda safe call*/
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


/* void SetUp_CUDA_Exe_Config() */
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


/*function to free memory*/
void dfree(double * arr[],int len)
{
        for(int i=0;i<len;i++)
                CUDA_SAFE_CALL(cudaFree(arr[i]));
        printf("mem freed\n");
}


/*calculate Gflops*/
double calculate_gflops(double &Tsec)
{
        //printf("time taken is %.8lf\n",Tsec);
        double gflops=(1.0e-9 * (( 2.0 * vlen )/Tsec));
        //printf("Gflops is \t%f\n",gflops);
        return gflops;

}



/*function to print on the screen*/
void print_on_screen(char * program_name,float tsec,double gflops,int size,int flag)//flag=1 if gflops has been calculated else flag =0
{
        printf("\n---------------%s----------------\n",program_name);
        printf("\tSIZE\t TIME_SEC\t Gflops\n");
        if(flag==1)
        printf("\t%d\t%f\t%lf\t",size,tsec,gflops);
        else
        printf("\t%d\t%lf\t%lf\t",size,"---","---");

}



/*get device count*/
int get_DeviceCount()
{
        int count;
        cudaGetDeviceCount(&count);
        return count;
}



extern "C"
float gpu_partial_sum_calculation(int i)
{
	init(vlen);
		// start=(cudaEvent_t)malloc(sizeof(cudaEvent_t));
		//stop=(cudaEvent_t)malloc(sizeof(cudaEvent_t));
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		routine(&vector[i-1]);
		printf("val calculated by device %d=%f\n",vector[i-1].deviceId,vector[i-1].retVal);	
		return vector[i-1].retVal;
}

	
void routine(void* givendata)
{
        Data *data = (Data*)givendata;
        int len = data->size;
        float *a,*b,*part_c;
        float *d_a,*d_b,*d_part_c;
        a=data->a;
        b=data->b;
        part_c = (float*)malloc(len*sizeof(float));
	float c;
        CUDA_SAFE_CALL(cudaSetDevice(data->deviceId));

        CUDA_SAFE_CALL(cudaMalloc((void**)&d_a,len*sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_b,len*sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_part_c,len*sizeof(float)));

        CUDA_SAFE_CALL(cudaMemcpy(d_a,a,len*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_b,b,len*sizeof(float),cudaMemcpyHostToDevice));


        dim3 threadsPerBlock(16,16);

        int numBlocks;
        if( len /256 == 0)
                numBlocks=1;
        else
                numBlocks = len/100;
        dim3 blocksPerGrid(numBlocks ,1);

        printf("Calling kernel on device: %d\n",data->deviceId);
	if(data->deviceId==1)
	{
		// start=(cudaEvent_t)malloc(sizeof(cudaEvent_t));
                //stop=(cudaEvent_t)malloc(sizeof(cudaEvent_t));
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

		cudaEventRecord(start,0);
	        vvmul<<<blocksPerGrid,threadsPerBlock>>>(len,d_a,d_b,d_part_c);
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		
		cudaEventElapsedTime(&elapsedTime,start,stop);
		Tsec=elapsedTime*1.0e-3;
		printf("\n\ntime taken by device 0 is:   %.8lf\n\n",Tsec);
		print_on_screen("VECTOR VECTOR MULTIPLICATION (MULTIGPU+MPI)",Tsec,0,vlen/2,1);

	}
	else
	{
		// start1=(cudaEvent_t)malloc(sizeof(cudaEvent_t));
                //stop1=(cudaEvent_t)malloc(sizeof(cudaEvent_t));
                cudaEventCreate(&start1);
                cudaEventCreate(&stop1);

        	cudaEventRecord(start1,0);
        	vvmul<<<blocksPerGrid,threadsPerBlock>>>(len,d_a,d_b,d_part_c);
        	cudaEventRecord(stop1,0);
        	cudaEventSynchronize(stop1);

		cudaEventElapsedTime(&elapsedTime1,start1,stop1);
		Tsec1=elapsedTime1*1.0e-3;
		printf("\n\ntime taken by device 1:   %.8lf\n\n",Tsec1);
		print_on_screen("VECTOR VECTOR MULTIPLICATION (MULTIGPU+MPI)",Tsec1,0,vlen/2,0);

	}

	if(cudaPeekAtLastError())
                printf("KERNEL ERROR: %s\t on device:%d\n",cudaGetErrorString(cudaPeekAtLastError()),data->deviceId);

        CUDA_SAFE_CALL(cudaMemcpy(part_c,d_part_c,len*sizeof(float),cudaMemcpyDeviceToHost)); // this line has problem because the part_c array size / allocation .

        int ind;
        for(ind=0;ind<len;ind++)
                c += part_c[ind];


        CUDA_SAFE_CALL(cudaFree(d_a));
        CUDA_SAFE_CALL(cudaFree(d_b));
        CUDA_SAFE_CALL(cudaFree(d_part_c));

        free(part_c);
        data->retVal=c;
	printf("Exiting from device :%d \n",data->deviceId);
}
void init(int size)
{

        int devCount;
        CUDA_SAFE_CALL(cudaGetDeviceCount(&devCount));
         if(devCount < 2)
        {
                printf("Atleast 2 GPU's are needed :%d\n",devCount);
                exit(0);
        }
	printf("devices available\n");

	int vlength=size;
        int ind;

        hVectA=(float*)malloc(vlen*sizeof(float));
        hVectB=(float*)malloc(vlen*sizeof(float));

        for(ind=0;ind < vlen;ind++)
        {

		hVectA[ind]=2;
		hVectB[ind]=2;
        }

        vector[0].deviceId      = 0;
        vector[0].size          =vlength/2;
        vector[0].a             =hVectA;
        vector[0].b             =hVectB;

        vector[1].deviceId      = 1;
        vector[1].size          =vlength/2;
        vector[1].a             =hVectA + vlength/2 ;
        vector[1].b             =hVectB + vlength/2 ;
	

}
extern "C"
void hfree()
{
	free(hVectA);
	free(hVectB);
	printf("host mem freed successfully\n");
}
                          
