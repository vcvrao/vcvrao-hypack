
/*******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                      October 15-18, 2013

  Example     :  CUBlasSVectMatMult.cu

  Objective   : Write a CUDA Program for Matrix Vector multiplication 
                using CUBLAS2 library function calls.

  Input       : None

  Output      : Execution time in seconds , Gflops achieved

  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

*************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>
#include<sys/time.h>

#include "cublas.h"
#define SIZE 1024
#define EPS 1.0e-15

int size = SIZE;

cudaEvent_t start,stop;
cudaError_t ret;
cublasStatus status;

double  *host_Mat,*host_Vect,*host_ResVect,*cpu_ResVect;
double  *device_Mat,*device_Vect,*device_ResVect;
int     vlength ,matRowSize , matColSize;
float  Tsec;
float   elapsedTime;

// checking GPU all kind of ERROR
#define CUBLAS_SAFE_CALL(call)					\
	status=call;						\
	if(status != CUBLAS_STATUS_SUCCESS)			\
	 { printf(" Error in CUBLAS call.Program terminating\n");\
	    exit(-1);						\
	 }					


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


/*mem error*/
void mem_error(char *arrayname, char *benchmark, int len, char *type)
{
        printf("\nMemory not sufficient to allocate for array %s\n\tBenchmark : %s  \n\tMemory requested = %d number of %s elements\n",arrayname, benchmark, len, type);
        exit(-1);
}

/* Get the number of GPU devices present on the host */
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

/*sequential mat vect multiplication*/
void CPU_MatVect()
{
	cpu_ResVect = (double *)malloc(matRowSize*sizeof(double));
	 if(cpu_ResVect==NULL)
                mem_error("host_ResVect","vectmatmul",matRowSize,"double");
	int i,j;
	for(i=0;i<matRowSize;i++)
	{cpu_ResVect[i]=0;
	for(j=0;j<matColSize;j++)
	cpu_ResVect[i]+=host_Mat[i+vlength*j]*host_Vect[j];
	}
}

/*function to calculate relative error*/
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

/*function to print value on screen*/
void print_on_screen(char * program_name,float tsec,double gflops,int size,int flag)//flag=1 if gflops has been calculated else flag =0
{
        printf("\n---------------%s----------------\n",program_name);
        printf("\tSIZE\t TIME_SEC\t Gflops\n");
        if(flag==1)
        printf("\t%d\t%f\t%lf\t",size,tsec,gflops);
        else
        printf("\t%d\t%lf\t%lf\t",size,"---","---");

}


void Check_ErrorDeviation_CPUGPU(double *host_MatX,double *cpu_MatX)
{	int i;
	for(i =0;i<matColSize;i++)
	{
	 printf("\n%lf %lf ",host_MatX[i],cpu_MatX[i]);
	if(abs(host_MatX[i]-cpu_MatX[i])>EPS)
	{
	printf("there is deviation in CPU and GPU result\n");
	return;
	}
	}
	 printf(" CPU and GPU results matched....");
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


/*function to calculate gflops*/
double calculate_gflops(float &Tsec)
{
//        printf("time taken is %.8lf\n",Tsec);
        float gflops=(1.0e-9 * (( 2.0 * size*size )/Tsec));
  //      printf("Gflops is \t%f\n",gflops);
        return gflops;
}





/*********************************************************************

          CUBLAS MAT-VECT MULTIPLICATION

**********************************************************************/

/* function to launch kernel*/
void launch_Cublas_dp_MatVect()
{
double alpha=1.0;
double beta=0.0;

cublasDgemv ('N', matRowSize, matColSize, alpha, device_Mat, matRowSize, device_Vect, 1, beta, device_ResVect, 1);

}

/*main function*/
int main()
{

	cudaDeviceProp deviceProp;
 	int device_Count=get_DeviceCount();
        printf("\n\nNUmber of Devices : %d\n\n", device_Count);

        // Device Selection, Device 1: Tesla C1060
        cudaSetDevice(0);

        int device;
        // Current Device Detection
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&deviceProp,device);
        printf("Using device %d: %s \n", device, deviceProp.name);


	// Vector length , Matrix Row and Col sizes..............
       	vlength = matColSize = size;
       	matRowSize = size;
    
	//printf("this programs does computation of square matrix only\n");

 	/*allocating the memory for each matrix */
        host_Mat =(double *)malloc(matRowSize*matColSize*sizeof(double));
        host_Vect = (double *)malloc(vlength*sizeof(double));
        host_ResVect = (double *)malloc(matRowSize*sizeof(double));

	
	// ---------------checking host memory  for error..............................
	if(host_Mat==NULL)
                mem_error("host_Mat","vectmatmul",matRowSize*matColSize,"double");

        if(host_Vect==NULL)
                mem_error("host_Vect","vectmatmul",vlength,"double");

        if(host_ResVect==NULL)
                mem_error("host_ResVect","vectmatmul",matRowSize,"double");
	


	//--------------Initializing the input arrays..............
        fill_dp_vector(host_Mat,matRowSize*matColSize);
        fill_dp_vector(host_Vect,vlength);


  	/* allocate memory for GPU events 
	start = (cudaEvent_t) malloc (sizeof(cudaEvent_t));
	stop = (cudaEvent_t) malloc (sizeof(cudaEvent_t));	
	if(start==NULL)
                mem_error("start","matmatmul",1,"cudaEvent_t");
        if(stop==NULL)
                mem_error("stop","matmatmul",1,"cudaEvent_t");*/
       	
	
  	//event creation...
	CUDA_SAFE_CALL(cudaEventCreate (&start));
        CUDA_SAFE_CALL(cudaEventCreate (&stop));

  	//allocating memory on GPU
	//	printf(" Allocating mem on gpu...");
	CUBLAS_SAFE_CALL(cublasAlloc (matRowSize*matColSize, sizeof(double), (void**)&device_Mat));
	CUBLAS_SAFE_CALL(cublasAlloc (vlength, sizeof(double), (void**)&device_Vect));
	CUBLAS_SAFE_CALL(cublasAlloc (matRowSize, sizeof(double), (void**)&device_ResVect));
 	
	// Initialization of vectors with host vectors 
	CUBLAS_SAFE_CALL(cublasSetVector (matRowSize*matColSize, sizeof(double), host_Mat, 1, device_Mat, 1));
	CUBLAS_SAFE_CALL(cublasSetVector (vlength, sizeof(double), host_Vect, 1, device_Vect, 1));

	// Launching CUBLAS call.........
	CUBLAS_SAFE_CALL(cublasGetError());
	
	CUDA_SAFE_CALL(cudaEventRecord (start, 0)); 
	launch_Cublas_dp_MatVect();//...........cublas_dgemv is called....

	CUDA_SAFE_CALL(cudaEventRecord (stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize (stop));
	CUBLAS_SAFE_CALL(cublasGetError());
	
	//retriving result from device
       	CUBLAS_SAFE_CALL(cublasGetVector (matRowSize, sizeof(double), device_ResVect, 1, host_ResVect, 1));

	//computing elapsed time	
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));

	Tsec = elapsedTime *1.0e-3; //time in sec now
	// calling funtion for measuring Gflops

        calculate_gflops(Tsec);

	//printing the result on screen
    	print_on_screen("CUBLAS MAT VECT MULTIPLICATION",Tsec,calculate_gflops(Tsec),size,1);


	// CPU calculation..and checking error deviation....
        CPU_MatVect();
  	relError(cpu_ResVect,host_ResVect,size);

	/*free the memory of CUBLAS */
	CUBLAS_SAFE_CALL(cublasFree(device_Mat));
	CUBLAS_SAFE_CALL(cublasFree(device_Vect));
	CUBLAS_SAFE_CALL(cublasFree(device_ResVect));
	
	// ending CUBLAS routines...
	CUBLAS_SAFE_CALL(cublasShutdown());

	/* Free the Host memory */
	free(host_Mat);
	free(host_Vect);
	free(host_ResVect);
	free(cpu_ResVect);
	return 0;
}// end of main
