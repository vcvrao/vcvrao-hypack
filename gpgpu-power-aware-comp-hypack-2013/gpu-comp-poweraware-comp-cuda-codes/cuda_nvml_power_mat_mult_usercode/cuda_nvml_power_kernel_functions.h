/**
 * This is a Header file corresponding to power_kernel.cu
 * It includes all prototypes & global variable declarations
**/

#include"cuda_nvml_power_kernel_define.h"
void dfree(double * arr[],int len);
int matMatMultCheckResultGMDP (double *hMatA, double *hMatB,double *output, int rows, int cols);
void checkBlockGridDim(cudaDeviceProp devProp,dim3 blockDim,dim3 gridDim);
void memErr(char *arrayname, char *benchmark, int len, char *numElements);
void printOnScreen(char * program_name,float tsec,double gFlops,int size,int flag);
double calGFlops(double &Tsec);
void kernelMatMult();



/**
 *   Host matrices 
**/
double *hMatA,*hMatB,*hMatC,*CPU_Result;

/**
 *   Device matrices 
**/
double *dMatA,*dMatB,*dMatC;

/**
 *   Size of a Matrix 
**/
int size = SIZE;

/**
 *   holds total time taken for execution 
**/
float elapsedTime;

/**
 *   Holds time
**/
double Tsec;

/**
 *   holds gflops
**/
double gFlops;

/**
 *   conditional variable 
**/
int sigFlag = 1;

/**
 *   cuda event objects
**/
cudaEvent_t start,stop;

/**
 *   device query objects
**/
cudaDeviceProp deviceProp;


