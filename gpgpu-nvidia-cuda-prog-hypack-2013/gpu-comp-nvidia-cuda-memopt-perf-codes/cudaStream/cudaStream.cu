/*
  STREAM benchmark implementation in CUDA.

    COPY:       a(i) = b(i)                 
    SCALE:      a(i) = q*b(i)               
    SUM:        a(i) = b(i) + c(i)          
    TRIAD:      a(i) = b(i) + q*c(i)        

  It measures the memory system on the device.
  The implementation is in single precision.

  Code based on the code developed by John D. McCalpin
  http://www.cs.virginia.edu/stream/FTP/Code/stream.c

  Written by: Massimiliano Fatica, NVIDIA Corporation
  Modified by: Douglas Enright (dpephd-nvidia@yahoo.com), 1 December 2010
  Extensive Revisions, 4 December 2010

  User interface motivated by bandwidthTest NVIDIA SDK example.
*/

#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>

#define N	8000000
#define NTIMES	10

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#define CUDA_SAFE_CALL(call){\
      cudaError_t err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(-1);                                                  \
    }}\



const double dbl_eps = 2.2204460492503131e-16;

__global__ void set_array(double *a,  double value, size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    a[idx] = value;
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Copy(double *a, double *b, size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    b[idx] = a[idx];
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Copy_Optimized(double *a, double *b, size_t len)
{
  /* 
   * Ensure size of thread index space is as large as or greater than 
   * vector index space else return;
   */
  if (blockDim.x * gridDim.x < len) return; 
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) b[idx] = a[idx];
}

__global__ void STREAM_Scale(double *a, double *b, double scale,  size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    b[idx] = scale* a[idx];
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Scale_Optimized(double *a, double *b, double scale,  size_t len)
{
  /* 
   * Ensure size of thread index space is as large as or greater than 
   * vector index space else return.
   */
  if (blockDim.x * gridDim.x < len) return; 
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) b[idx] = scale* a[idx];
}

__global__ void STREAM_Add( double *a, double *b, double *c,  size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    c[idx] = a[idx]+b[idx];
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Add_Optimized( double *a, double *b, double *c,  size_t len)
{
  /* 
   * Ensure size of thread index space is as large as or greater than 
   * vector index space else return.
   */
  if (blockDim.x * gridDim.x < len) return; 
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) c[idx] = a[idx]+b[idx];
}

__global__ void STREAM_Triad( double *a, double *b, double *c, double scalar, size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    c[idx] = a[idx]+scalar*b[idx];
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Triad_Optimized( double *a, double *b, double *c, double scalar, size_t len)
{
  /* 
   * Ensure size of thread index space is as large as or greater than 
   * vector index space else return.
   */
  if (blockDim.x * gridDim.x < len) return; 
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) c[idx] = a[idx]+scalar*b[idx];
}

/* Host side verification routines */
bool STREAM_Copy_verify(double *a, double *b, size_t len) {
  size_t idx;
  bool bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    double expectedResult = a[idx];
    double diffResultExpected = (b[idx] - expectedResult);
    double relErrorULPS = (fabs(diffResultExpected)/fabs(expectedResult))/dbl_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 2.);
  }  

  return bDifferent;
}

bool STREAM_Scale_verify(double *a, double *b, double scale, size_t len) {
  size_t idx;
  bool bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    double expectedResult = scale*a[idx];
    double diffResultExpected = (b[idx] - expectedResult);
    double relErrorULPS = (fabs(diffResultExpected)/fabs(expectedResult))/dbl_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 2.);
  }  

  return bDifferent;
}

bool STREAM_Add_verify(double *a, double *b, double *c, size_t len) {
  size_t idx;
  bool bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    double expectedResult = a[idx] + b[idx];
    double diffResultExpected = (c[idx] - expectedResult);
    double relErrorULPS = (fabs(diffResultExpected)/fabs(expectedResult))/dbl_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 2.);
  }

  return bDifferent;
}

bool STREAM_Triad_verify(double *a, double *b, double *c, double scalar, size_t len) {
  size_t idx;
  bool bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    double expectedResult = a[idx] + scalar*b[idx];
    double diffResultExpected = (c[idx] - expectedResult);
    double relErrorULPS = (fabs(diffResultExpected)/fabs(expectedResult))/dbl_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 3.);
  }

  return bDifferent;
}

/* forward declarations */
int setupStream(const int argc, const char **argv);
void runStream(const int iNumThreadsPerBlock, bool bDontUseGPUTiming);
void printResultsReadable(float times[][NTIMES]);
void printHelp(void);

int main(int argc, char *argv[])
{
 
  printf("[Double-Precision Device-Only STREAM Benchmark implementation in CUDA]\n");

  //set logfile name and start logs
  printf("streamBenchmark.txt");
  printf("%s Starting...\n\n", argv[0]);

  int iRetVal = setupStream(argc, (const char**)argv);
  if (iRetVal != -1)
  {
    printf("\n[streamBenchmark] - results:\t%s\n\n", (iRetVal == 0) ? "PASSES" : "FAILED");
  }
}

///////////////////////////////////////////////////////////////////////////////
//Parse args, run the appropriate tests
///////////////////////////////////////////////////////////////////////////////
int setupStream(const int argc, const char **argv)
{
  int deviceNum = 0;
  //char *device = NULL;
  bool bDontUseGPUTiming = false;
  int iNumThreadsPerBlock = 128;


  cudaSetDevice(deviceNum);
  cudaDeviceProp deviceProp;
  if (cudaGetDeviceProperties(&deviceProp, deviceNum) == cudaSuccess) {
    printf(" Device %d: %s\n", deviceNum, deviceProp.name);
  } else {
    printf(" Unable to determine device %d properties, exiting\n");
    return -1;
  }

  if (deviceProp.major == 1 && deviceProp.minor < 3) {
    printf(" Unable to run double-precision STREAM benchmark on a compute capability GPU less than 1.3\n");
    return -1;
  }

  if (deviceProp.major == 2 && deviceProp.minor == 1) {
    iNumThreadsPerBlock = 192; /* GF104 architecture / 48 CUDA Cores per MP */
  } else {
    iNumThreadsPerBlock = 128; /* GF100 architecture / 32 CUDA Cores per MP */
  }	

  CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceBlockingSync));

  /*if(shrCheckCmdLineFlag( argc, argv, "cputiming")) {
    bDontUseGPUTiming = true;
    printf(" Using cpu-only timer.\n");
  }*/
	
  runStream(iNumThreadsPerBlock, bDontUseGPUTiming);

  return 0;
}

///////////////////////////////////////////////////////////////////////////
// runStream
///////////////////////////////////////////////////////////////////////////
void runStream(const int iNumThreadsPerBlock, bool bDontUseGPUTiming)
{
  double *d_a, *d_b, *d_c;

  int k;
  float times[8][NTIMES];
  double scalar;

  /* Allocate memory on device */
  CUDA_SAFE_CALL( cudaMalloc((void**)&d_a, sizeof(double)*N) );
  CUDA_SAFE_CALL( cudaMalloc((void**)&d_b, sizeof(double)*N) );
  CUDA_SAFE_CALL( cudaMalloc((void**)&d_c, sizeof(double)*N) );

  /* Compute execution configuration */
  dim3 dimBlock(iNumThreadsPerBlock); /* (iNumThreadsPerBlock,1,1) */
  dim3 dimGrid(N/dimBlock.x); /* (N/dimBlock.x,1,1) */
  if( N % dimBlock.x != 0 ) dimGrid.x+=1; 

  printf(" Array size (double precision) = %u\n",N);
  printf(" using %u threads per block, %u blocks\n",dimBlock.x,dimGrid.x);

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_a, 2., N);
  set_array<<<dimGrid,dimBlock>>>(d_b, .5, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5, N);

  /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */
  cudaEvent_t start, stop;

  /* both timers report msec */
  CUDA_SAFE_CALL( cudaEventCreate( &start ) );  /* gpu timer facility */
  CUDA_SAFE_CALL( cudaEventCreate( &stop ) );   /* gpu timer facility */

  scalar=3.0;
  for (k=0; k<NTIMES; k++)
  {

    CUDA_SAFE_CALL( cudaEventRecord( start, 0 ) );
    STREAM_Copy<<<dimGrid,dimBlock>>>(d_a, d_c, N); 
    CUDA_SAFE_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
    //get the the total elapsed time in ms
      CUDA_SAFE_CALL( cudaEventElapsedTime( &times[0][k], start, stop ) );
 
    CUDA_SAFE_CALL( cudaEventRecord( start, 0 ) );
    STREAM_Copy_Optimized<<<dimGrid,dimBlock>>>(d_a, d_c, N); 
    CUDA_SAFE_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
    //get the the total elapsed time in ms
      CUDA_SAFE_CALL( cudaEventElapsedTime( &times[1][k], start, stop ) );
 
    CUDA_SAFE_CALL( cudaEventRecord( start, 0 ) );
    STREAM_Scale<<<dimGrid,dimBlock>>>(d_b, d_c, scalar,  N);
    CUDA_SAFE_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
    //get the the total elapsed time in ms
      CUDA_SAFE_CALL( cudaEventElapsedTime( &times[2][k], start, stop ) );

    CUDA_SAFE_CALL( cudaEventRecord( start, 0 ) );
    STREAM_Scale_Optimized<<<dimGrid,dimBlock>>>(d_b, d_c, scalar,  N);
    CUDA_SAFE_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
    //get the the total elapsed time in ms
      CUDA_SAFE_CALL( cudaEventElapsedTime( &times[3][k], start, stop ) );

    CUDA_SAFE_CALL( cudaEventRecord( start, 0 ) );
    STREAM_Add<<<dimGrid,dimBlock>>>(d_a, d_b, d_c,  N);
    CUDA_SAFE_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
    //get the the total elapsed time in ms
      CUDA_SAFE_CALL( cudaEventElapsedTime( &times[4][k], start, stop ) );

    CUDA_SAFE_CALL( cudaEventRecord( start, 0 ) );
    STREAM_Add_Optimized<<<dimGrid,dimBlock>>>(d_a, d_b, d_c,  N);
    CUDA_SAFE_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
    //get the the total elapsed time in ms
      CUDA_SAFE_CALL( cudaEventElapsedTime( &times[5][k], start, stop ) );

    CUDA_SAFE_CALL( cudaEventRecord( start, 0 ) );
    STREAM_Triad<<<dimGrid,dimBlock>>>(d_b, d_c, d_a, scalar,  N);
    CUDA_SAFE_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
    //get the the total elapsed time in ms
      CUDA_SAFE_CALL( cudaEventElapsedTime( &times[6][k], start, stop ) );

    CUDA_SAFE_CALL( cudaEventRecord( start, 0 ) );
    STREAM_Triad_Optimized<<<dimGrid,dimBlock>>>(d_b, d_c, d_a, scalar,  N);
    CUDA_SAFE_CALL( cudaEventRecord( stop, 0 ) );
    CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
    //get the the total elapsed time in ms
      CUDA_SAFE_CALL( cudaEventElapsedTime( &times[7][k], start, stop ) );

  }

  /* verify kernels */
  double *h_a, *h_b, *h_c;
  bool errorSTREAMkernel = true;

  if ( (h_a = (double*)calloc( N, sizeof(double) )) == (double*)NULL ) {
    printf("Unable to allocate array h_a, exiting ...\n");
    exit(1);
  }
  if ( (h_b = (double*)calloc( N, sizeof(double) )) == (double*)NULL ) {
    printf("Unable to allocate array h_b, exiting ...\n");
    exit(1);
  }

  if ( (h_c = (double*)calloc( N, sizeof(double) )) == (double*)NULL ) {
    printf("Unalbe to allocate array h_c, exiting ...\n");
    exit(1);
  }

  /* 
   * perform kernel, copy device memory into host memory and verify each 
   * device kernel output 
   */

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);

  STREAM_Copy<<<dimGrid,dimBlock>>>(d_a, d_c, N); 
  CUDA_SAFE_CALL( cudaMemcpy( h_a, d_a, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy( h_c, d_c, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  errorSTREAMkernel = STREAM_Copy_verify(h_a, h_c, N);
  if (errorSTREAMkernel) {
    printf(" device STREAM_Copy:\t\t\tError detected in device STREAM_Copy, exiting\n");
    exit(-2000);
  } else {
    printf(" device STREAM_Copy:\t\t\tPass\n"); 
  }

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);
  
  STREAM_Copy_Optimized<<<dimGrid,dimBlock>>>(d_a, d_c, N); 
  CUDA_SAFE_CALL( cudaMemcpy( h_a, d_a, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy( h_c, d_c, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  errorSTREAMkernel = STREAM_Copy_verify(h_a, h_c, N);
  if (errorSTREAMkernel) {
    printf(" device STREAM_Copy_Optimized:\t\tError detected in device STREAM_Copy_Optimized, exiting\n");
    exit(-3000);
  } else {
    printf(" device STREAM_Copy_Optimized:\t\tPass\n"); 
  }

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);
  
  STREAM_Scale<<<dimGrid,dimBlock>>>(d_b, d_c, scalar, N); 
  CUDA_SAFE_CALL( cudaMemcpy( h_b, d_b, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy( h_c, d_c, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  errorSTREAMkernel = STREAM_Scale_verify(h_b, h_c, scalar, N);
  if (errorSTREAMkernel) {
    printf(" device STREAM_Scale:\t\t\tError detected in device STREAM_Scale, exiting\n");
    exit(-4000);
  } else {
    printf(" device STREAM_Scale:\t\t\tPass\n");
  }

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);
  
  STREAM_Scale_Optimized<<<dimGrid,dimBlock>>>(d_b, d_c, scalar, N); 
  CUDA_SAFE_CALL( cudaMemcpy( h_b, d_b, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy( h_c, d_c, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  errorSTREAMkernel = STREAM_Scale_verify(h_b, h_c, scalar, N);
  if (errorSTREAMkernel) {
    printf(" device STREAM_Scale_Optimized:\t\tError detected in device STREAM_Scale_Optimized, exiting\n");
    exit(-5000);
  } else {
    printf(" device STREAM_Scale_Optimized:\t\tPass\n");
  }

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);

  STREAM_Add<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, N); 
  CUDA_SAFE_CALL( cudaMemcpy( h_a, d_a, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy( h_b, d_b, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy( h_c, d_c, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  errorSTREAMkernel = STREAM_Add_verify(h_a, h_b, h_c, N);
  if (errorSTREAMkernel) {
    printf(" device STREAM_Add:\t\t\tError detected in device STREAM_Add, exiting\n");
    exit(-6000);
  } else {
    printf(" device STREAM_Add:\t\t\tPass\n");
  }

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);

  STREAM_Add_Optimized<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, N); 
  CUDA_SAFE_CALL( cudaMemcpy( h_a, d_a, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy( h_b, d_b, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy( h_c, d_c, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  errorSTREAMkernel = STREAM_Add_verify(h_a, h_b, h_c, N);
  if (errorSTREAMkernel) {
    printf(" device STREAM_Add_Optimized:\t\tError detected in device STREAM_Add_Optimzied, exiting\n");
    exit(-7000);
  } else {
    printf(" device STREAM_Add_Optimzied:\t\tPass\n");
  }

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);

  STREAM_Triad<<<dimGrid,dimBlock>>>(d_b, d_c, d_a, scalar, N); 
  CUDA_SAFE_CALL( cudaMemcpy( h_a, d_a, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy( h_b, d_b, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy( h_c, d_c, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  errorSTREAMkernel = STREAM_Triad_verify(h_b, h_c, h_a, scalar, N);
  if (errorSTREAMkernel) {
    printf(" device STREAM_Triad:\t\t\tError detected in device STREAM_Triad, exiting\n");
    exit(-8000);
  } else {
    printf(" device STREAM_Triad:\t\t\tPass\n");
  }

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);

  STREAM_Triad_Optimized<<<dimGrid,dimBlock>>>(d_b, d_c, d_a, scalar, N); 
  CUDA_SAFE_CALL( cudaMemcpy( h_a, d_a, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy( h_b, d_b, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy( h_c, d_c, sizeof(double) * N, cudaMemcpyDeviceToHost) );
  errorSTREAMkernel = STREAM_Triad_verify(h_b, h_c, h_a, scalar, N);
  if (errorSTREAMkernel) {
    printf(" device STREAM_Triad_Optimized:\t\tError detected in device STREAM_Triad_Optimized, exiting\n");
    exit(-9000);
  } else {
    printf(" device STREAM_Triad_Optimized:\t\tPass\n");
  }

  /* continue from here */
  printResultsReadable(times);
 
  //clean up timers
  CUDA_SAFE_CALL(cudaEventDestroy( stop ) );
  CUDA_SAFE_CALL(cudaEventDestroy( start ) );
 
  /* Free memory on device */
  CUDA_SAFE_CALL( cudaFree(d_a) );
  CUDA_SAFE_CALL( cudaFree(d_b) );
  CUDA_SAFE_CALL( cudaFree(d_c) );

}

///////////////////////////////////////////////////////////////////////////
//Print Results to Screen and File
///////////////////////////////////////////////////////////////////////////
void printResultsReadable(float times[][NTIMES]) {

  int j,k;

  float	avgtime[8] = {0., 0., 0., 0., 0., 0., 0., 0.};
  float maxtime[8] = {0., 0., 0., 0., 0., 0., 0., 0.};
  float mintime[8] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

  char	*label[8] = {"Copy:      ", "Copy Opt:  ", "Scale:     ", "Scale Opt: ", "Add:       ", "Add Opt:   ", "Triad:     ", "Triad Opt: "};

  float	bytes_per_kernel[8] = { 
    2. * sizeof(double) * N, /* Copy */
    2. * sizeof(double) * N, /* Copy Opt */
    2. * sizeof(double) * N, /* Scale */
    2. * sizeof(double) * N, /* Scale Opt */
    3. * sizeof(double) * N, /* Add */
    3. * sizeof(double) * N, /* Add Opt */
    3. * sizeof(double) * N, /* Triad */
    3. * sizeof(double) * N  /* Triad Opt */
  }; 

  /* --- SUMMARY --- */

  for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
  {
    for (j=0; j<8; j++)
    {
      avgtime[j] = avgtime[j] + (1.e-03f * times[j][k]);
      mintime[j] = MIN(mintime[j], (1.e-03f * times[j][k]));
      maxtime[j] = MAX(maxtime[j], (1.e-03f * times[j][k]));
    }
  }
 
  printf("Function    Rate (MB/s)    Avg time      Min time      Max time\n");
  
  for (j=0; j<8; j++) {
     avgtime[j] = avgtime[j]/(float)(NTIMES-1);
     
     printf("%s%11.4f  %11.6f  %11.6f  %11.6f\n", label[j], 1.0E-06 * bytes_per_kernel[j]/mintime[j], avgtime[j], mintime[j], maxtime[j]);
  }
  
}

///////////////////////////////////////////////////////////////////////////
//Print help screen
///////////////////////////////////////////////////////////////////////////
void printHelp(void)
{
  printf("Usage:  streamdp [OPTION]...\n");
  printf("Double-Pecision STREAM Benchmark implementation in CUDA\n");
  printf("Performs Copy, Scale, Add, and Triad double-precision kernels\n");
  printf("\n");
  printf("Example: ./streamdp\n");
  printf("\n");
  printf("Options:\n");
  printf("--help\t\t\tDisplay this help menu\n");
  printf("--device=[deviceno]\tSpecify the device to be used (Default: device 0)\n");
  printf("--cputiming\t\tForce CPU-based timing to be used\n");
}
