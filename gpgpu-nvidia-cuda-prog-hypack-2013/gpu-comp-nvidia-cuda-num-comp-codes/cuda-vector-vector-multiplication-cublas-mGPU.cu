

/*********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example     : cuda-vector-vector-multiplication-cublas-mGPU.cu

  Objective   : Write a CUDA  program to compute Vector-Vector multiplication  
                using CUBLAS library function calls to be executed on multiple GPUs.

  Input       : None

  Output      : Execution time in seconds , Gflops achieved

  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

****************************************************************************/

#include<stdio.h>
#include<cuda.h>
#include<pthread.h>
#include<error.h>
#include "cublas.h"

#define EPS 1.0e-12
#define GRIDSIZE 10
#define BLOCKSIZE 16

#define SIZE 128

struct Data
{
	int deviceId;
	int size;
	double* a;
	double* b;
	double retVal;
};
cublasStatus status;
#define CUBLAS_SAFE_CALL(call)                                  \
        status=call;                                            \
        if(status != CUBLAS_STATUS_SUCCESS)                     \
         { printf(" Error in CUBLAS call.Program terminating\n");\
            exit(-1);                                           \
         }                            


int size=SIZE;

double calculate_gflops(float &Tsec)
{
 double gflops;
	gflops=(1.0e-9 * (( 2.0 * size )/Tsec));

	return gflops;
}

/*
 * Check for safe return of all calls to the device
 */


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

/* prints the result in screen */
void print_on_screen(char * program_name,float tsec,double gflops,int size,int flag)//flag=1 if gflops has been calculated else flag =0
{
        printf("\n---------------%s----------------\n",program_name);
        printf("\tSIZE\t TIME_SEC\t Gflops\n");
        if(flag==1)
        printf("\t%d\t%f\t%lf\t",size,tsec,gflops);
        else
        printf("\t%d\t%lf\t%lf\t",size,"---","---");

}

/* Function to do computations on GPU */
void* routine(void* givendata)
{
	Data *data = (Data*)givendata;
	int len = data->size;
	double *a,*b;
	double *d_a,*d_b;
	double c;

	a=data->a;
	b=data->b;

	cudaDeviceProp deviceProp;	
	float elapsedTime,Tsec;
	cudaEvent_t start,stop;

	CUDA_SAFE_CALL(cudaSetDevice(data->deviceId));
	cudaGetDeviceProperties(&deviceProp,data->deviceId);
        printf("Using device %d: %s \n", data->deviceId, deviceProp.name);
	
	/* allocate memory for GPU events 
        start = (cudaEvent_t) malloc (sizeof(cudaEvent_t));
        stop = (cudaEvent_t) malloc (sizeof(cudaEvent_t));
        if(start==NULL)
                mem_error("start","vectvectmul",1,"cudaEvent_t");
        if(stop==NULL)
                mem_error("stop","vectvectmul",1,"cudaEvent_t");*/

  	/*  event creation */
        CUDA_SAFE_CALL(cudaEventCreate (&start));
        CUDA_SAFE_CALL(cudaEventCreate (&stop));

	
	CUBLAS_SAFE_CALL( cublasInit());

	CUBLAS_SAFE_CALL(cublasAlloc (len, sizeof(double), (void**)&d_a));
        CUBLAS_SAFE_CALL(cublasAlloc (len, sizeof(double), (void**)&d_b));

	CUBLAS_SAFE_CALL(cublasSetVector (len, sizeof(double), a, 1,d_a, 1));
        CUBLAS_SAFE_CALL(cublasSetVector (len, sizeof(double), b, 1,d_b, 1));

	CUDA_SAFE_CALL(cudaEventRecord (start, 0));

	c= cublasDdot (len, d_a, 1, d_b, 1);	

        CUDA_SAFE_CALL(cudaEventRecord (stop, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize (stop));
        CUDA_SAFE_CALL(cudaEventElapsedTime ( &elapsedTime, start, stop));

        Tsec= 1.0e-3*elapsedTime;

	/* printing the result on screen */
    	print_on_screen("MULTI_GPU_CUBLAS VECT VECT MULTIPLICATION",Tsec,calculate_gflops(Tsec),len,1);


	data->retVal=c;
		
	
	CUBLAS_SAFE_CALL(cublasFree(d_a));
        CUBLAS_SAFE_CALL(cublasFree(d_b));
        CUBLAS_SAFE_CALL( cublasShutdown());
	return 0;
}

/*  Function for CPU vector-vector multiplication */
void compare(double *v1,double *v2,double *hres,int len)
{
	int i;
	*hres=0.00;
	for(i=0;i<len;i++)
		*hres+=v1[i]*v2[i];
}

/* Fill in the vector with double precision values */
void fill_dp_vector(double* vec,int size)
{
        int ind;
        for(ind=0;ind<size;ind++)
                vec[ind]=drand48();
}

/* Compare CPU and GPU results  */
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

int main()
{	

	int devCount;
	CUDA_SAFE_CALL(cudaGetDeviceCount(&devCount));
	printf("\n\nNUmber of Devices : %d\n\n", devCount);

	if(devCount < 2)
	{
		printf("Atleast 2 GPU's are needed :%d\n",devCount);
		exit(0);
	}

	double *hVectA,*hVectB,hRes=0,cpuRes=0;
	int vlen=size;

	hVectA=(double*)malloc(vlen*sizeof(double));
	hVectB=(double*)malloc(vlen*sizeof(double));

	
	/* Initializing the input arrays */
        fill_dp_vector(hVectA,vlen);
        fill_dp_vector(hVectB,vlen);


	Data vector[2];
	
	vector[0].deviceId 	= 0;
	vector[0].size		=vlen/2;
	vector[0].a		=hVectA;
	vector[0].b		=hVectB;	   
	
	vector[1].deviceId 	= 1;
	vector[1].size		=vlen-vlen/2;
	vector[1].a		=hVectA + vlen/2 ;
	vector[1].b		=hVectB + vlen/2 ;	   

	
	pthread_t thread;
	if(pthread_create(&thread,NULL,routine,(void*)&vector[0]) != 0)
		perror("Thread creation error\n");
	routine(&vector[1]);
	pthread_join(thread,NULL);

	hRes=vector[0].retVal + vector[1].retVal;


	compare(hVectA,hVectB,&cpuRes,vlen);

	relError(&cpuRes,&hRes,1);

	free(hVectA);
	free(hVectB);
	
	return 0;
}
