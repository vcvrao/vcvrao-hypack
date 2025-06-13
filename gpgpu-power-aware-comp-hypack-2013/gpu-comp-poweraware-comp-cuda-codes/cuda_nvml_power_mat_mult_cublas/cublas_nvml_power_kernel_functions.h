/**
 * This header file is specially for calculatin kernel 
 * "matrix matrix multiplication using vendor supplied library (cublas)".
**/ 

/**
 * include files 
**/
#include<cublas_nvml_power_kernel_define.h>
#include<cublas.h>


#define MAT_START_SIZE 10240 
#define LENGTH 60
#define ERRORFILE "./errorFile"

/**
 * conditional variable 
**/
int sigFlag = 1;

/**
 * calculate elapsed time between calculation 
 * of matrix multipliaction. 
**/
float           *elapsedTime; 

/**
 * holds value of flops 
**/
double          mflops, gflops; 

/**
 * holds value of total time taken to execute a kernel. 
**/
double          Tsec_gpu=0.0; 

/**
 * holds matrix dimension and leading dimensions.
**/
int             M, K, N, i,  n_gpu, lda, ldb, ldc; 

/**
 * holds total number of GPU event to be recorded.
**/
#define TOTALEVENT 5  

#define LINE_DOT "\n.....................................................................................................\n"
#define LINE "\n__________________________________________________________________________________________________________\n"

/**
 *  function prototypes for fucntion included in 
 *  kenrel calculation file. 
**/
char* getGPUDevName(int GPUDevId);
int print_error(char *msg,int nodeNum, int devNum , char *benchName);
int checkResult(double *InMatA, double *InMatB, double *outMatC, int m, int n , int k );
int MatMatMult(int deviceNum, int GPUDevId, int );

