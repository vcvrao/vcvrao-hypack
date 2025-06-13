/**************************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

Descritption    : The Simple example program to demonstart Matrix Matrix Multiplication 
                  using (OpenCL+ OpenMP )Multi GPU and Device Global Memory. 

                  Host reads the two matrices A & B . The matrix A is divided among the 
                  number of devices by row wise partitioning and that chunk of data will
                  be copied from host to each device. Each GPU device perform multiplication  
                  on its assigned rows. The Host reads the result from each device and 
                  accumulate.
  
Input           : Matrix Size

Output          : Time taken for computation

Created         : August-2013

E-mail          : hpcfte@cdac.in     

***************************************************************************************/

/* Header file inclusion */
#include<CL/cl.h>
#include<assert.h>
#include<stdio.h>
#include<string.h>
#include<omp.h>
#include<math.h>

#define EPS 1.0e-8f
#define KB 1024.0

extern "C" void fillArray (float * matrix, int rows, int cols );
extern "C" void setExeEnv ( cl_context *context, cl_uint *num_devices, cl_device_id **devices, cl_program *program);
extern "C" void matMatMultiplication (cl_uint num_devices,cl_device_id *devices, cl_program program,cl_context context,float * h_a, float *h_b, float *h_output);
extern "C" int  checkResult (float *h_a, float *h_b,float *output, int rows, int cols);
extern "C" void print(float *h_a, float *h_b, float *h_output);
extern "C" int  getPlatform(cl_platform_id *selected_platform);
extern "C" int  getDevices( cl_platform_id selected_platform, cl_uint *total_devices, cl_device_id **devices);
extern "C" void checkStatus(const char *name,cl_int err);
extern "C" char *readKernelSource(const char *kernel_source_path);
extern "C" void checkStatus(const char *name,cl_int err);
extern "C" int  getDeviceInfo(cl_device_id device);

/* define the  Kernel source path */
const char * kernel_source_path = "Mat_Mat_Mult_Openmp_Opencl_GlobalMemory.cl";

/* Global variables for Matrix dimentions */
long int matrixSize,height_a,width_a,height_b,width_b;

/****************************************************************
Function to return the execution time in seconds
****************************************************************/
double executionTime(cl_event &event)
{
    cl_ulong start, end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}

/****************************************************************
main() function
****************************************************************/
int main(int argc, char **argv)
{

	cl_uint		num_devices; /*hold the number of devices */
	cl_device_id	*devices; /* hold list of devices */	
	cl_context	context; /* hold the context */
	cl_program	program; /* hold the program object */
	long int 	rows, cols; /* hold the number of rows & cols of matrices */
	float 		*h_a, *h_b, *h_output; /* Host input / output matrices */
	int 		row_count, col_count; /* loop control variables */

	/** 1. Reading the input ***/
	if( argc != 2 ) {

                        printf("\n Usage : ./<executable> <matrix-size> \n");
                        return 0;
         }

	matrixSize= atoi(argv[1]);
        rows = cols = matrixSize;

	 height_a = width_b = rows;
         width_a = height_b = cols;


          /* 2. Allocating Memory for Host Input matrices  */
          assert( (h_a=(float *) malloc (sizeof(float) * height_a * width_a  )) != NULL) ;
          assert( (h_b =(float *) malloc (sizeof(float ) * height_b * width_b )) != NULL) ;
          assert( (h_output=(float *) malloc (sizeof(float) * height_a * width_b  )) != NULL) ;

          /* 3. Initializing Host Input matrices  */
          fillArray ( h_a, height_a, width_a);
          fillArray ( h_b, height_b, width_b);

          /* 4. Initializing Host output matrix with zero  */
           for(row_count=0; row_count < height_a; row_count++) {
                	for(col_count=0; col_count < width_b; col_count++) {
                           	h_output[row_count * width_b + col_count] = 0.00 ;
            		}
   	   }

	 /** Step 5-9 . Setting the exeution environment for openCL kernel**/
	 setExeEnv ( &context , &num_devices, &devices, &program );

	 printf("\n\n SUMMARY : ");
	 printf("\n _____________________________________________________________________________________________"); 
         printf( "\n Matrix Dimensions\t\t\t:  A( %d x %d ) , B( %d x %d ) , C(%d x %d)",height_a,width_a,height_b,
		     width_b, height_a , width_b);
	
	 /** Step 10-21. function to perform matrix multiplication on GPU **/
	matMatMultiplication (num_devices, devices, program, context, h_a, h_b, h_output);


  	 /*******************************************
        Un-comment below sections to print 
         --- Input & output matrices 
        *******************************************/
/*	
         print(h_a, h_b, h_output); 
*/	

        /* 22. Check the GPU results against CPU result **/
        if ( checkResult( h_a,h_b,h_output, height_a , width_a ))
                printf(" \n\t\t >> ERROR : Results are not Same from GPU & CPU compution \n");
        else
		printf("\n Result Verification : Success  \n\n");


	/** cleanup **/
	if( h_a)	free(h_a);
	if( h_b)	free(h_b);
	if( h_output)	free(h_output);

	clReleaseProgram( program );
	clReleaseContext( context );

}/* End of main() */


/*****************************************************************************************
*Function for getting the OpenCL Platform ID
*****************************************************************************************/
int getPlatform(cl_platform_id *selected_platform)
{
        cl_int          err;
        int             count;
        char            pbuff[100];
        cl_uint         num_platforms;
        cl_platform_id  *platforms;
        cl_uint 	numDevices;

        *selected_platform = NULL;

        /*  Get the number of OpenCL Platforms Available */
        err = clGetPlatformIDs ( 0, 0, &num_platforms);
        if( err != CL_SUCCESS || num_platforms == 0) {
                printf(" \n\t\t No Platform Found \n");
                return 1;
        }
        else {
                if( num_platforms == 0) {
                        printf(" \n\t\t No Platform Found \n");
                        return 1;
                }

                else {

                        /* Allocate the space for available platform*/
                        assert( (platforms = (cl_platform_id *) malloc( sizeof(cl_platform_id) * num_platforms)) != NULL);

                        /*  Get available OpenCL Platforms IDs*/
                        err = clGetPlatformIDs( num_platforms, platforms, NULL);
                        checkStatus(" Failed to get Platform IDs",err);

                        for ( count = 0 ; count < num_platforms ; count++) {

				/* get platform info*/
                                err=clGetPlatformInfo(platforms[count],CL_PLATFORM_NAME,sizeof(pbuff),pbuff,NULL);
                                checkStatus("clGetPlatformInfo Failed",err);
                                /* get device id and info*/
                                err = clGetDeviceIDs( platforms[count],CL_DEVICE_TYPE_GPU,0,0,&numDevices);
                                if( err != CL_SUCCESS  || numDevices ==0)
                               {
                                         continue;
                                }
                                else
                                {
                                        /* get selected platform*/
                                        *selected_platform =platforms[count];
                                        printf("\n Platform :  %s\n",pbuff);
                                        break;
                                }


                        }

                        free(platforms);
                }
        }
        return 0;
}

int getDeviceInfo( cl_device_id device)
{

        int                     icount;
        char                    dbuff[100];
        cl_uint                 device_add_bits;
        cl_uint                 device_max_comp_unit;   /* Device maximum compute units */
        cl_uint                  device_max_freq;        /* Device maximum clock frequency */
        cl_device_type           type;
        cl_int                   err;
        cl_ulong                 device_global_mem;        /* Hold Device memory size */
        cl_ulong                 device_cache_size;        /* Hold Device memory size */
        cl_ulong                 device_local_mem;        /* Hold Device memory size */
        cl_device_mem_cache_type device_cache_type; /* Hold the device cache type like read / write etc*/



       /* Get device Name */
       err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dbuff), &dbuff, NULL);
       checkStatus("clGetDeviceInfo Failed ",err);
       printf(" Device Used :  %s   ",dbuff);

       /* Get device type  */
       err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
       checkStatus("clGetDeviceInfo Failed ",err);
       switch ( type )
       {
              case CL_DEVICE_TYPE_CPU : printf("\n Device Type\t\t\t : CPU ");
              break;
              case CL_DEVICE_TYPE_GPU : printf("\n  Device Type\t: GPU");
              break;
              case CL_DEVICE_TYPE_ACCELERATOR:printf("\n Device Type\t\t\t : Dedicated OpenCLAccelerator");
              break;
              case CL_DEVICE_TYPE_DEFAULT : printf("\n Device Type\t\t\t: DEFAULT  ");
              break;
        }


        /* Get device  global memory in bytes */
         err = clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_global_mem),
                                &device_global_mem, NULL);
          checkStatus("clGetDeviceInfo Failed ",err);
          printf("\n  Global Memory\t: %lf MB  ",(double)(device_global_mem)/(KB * KB));

          /* Get device cache type */
          err = clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                                 sizeof(device_cache_type),
                                  &device_cache_type, NULL);
          checkStatus("clGetDeviceInfo Failed ",err);

          switch ( device_cache_type )
          {
                case CL_NONE : printf(" \( Cache Type : None ");
                break;
                case CL_READ_ONLY_CACHE : printf(" \( Cache Type: ro  ");
                break;
                case CL_READ_WRITE_CACHE : printf(" \( Cache Type : rw  ");
                break;
                                                                              
 }

         /* Get device cache size */
         err = clGetDeviceInfo(  device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                                   sizeof(device_cache_size),
                                   &device_cache_size, NULL);
         checkStatus("clGetDeviceInfo Failed ",err);
         printf(" , Cache Size : %ld Bytes ) ",device_cache_size);

          /* Get device local memory size in bytes */
          err = clGetDeviceInfo( device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_local_mem),
                                 &device_local_mem, NULL);
          checkStatus("clGetDeviceInfo Failed ",err);
          printf(" \n  Local Memory \t :%lf KB  ",(double)device_local_mem/(KB));


        return 0 ;

}
//////////////////////////////////////////////////////////////////////////////////////////
//Function for reading the kernel source  
//////////////////////////////////////////////////////////////////////////////////////////
char    *readKernelSource(const char *kernel_source_path)
{
        // local variable 
        FILE    *fp = NULL;
        size_t  source_length;
        char    *source_string ;

        fp = fopen( kernel_source_path , "r");
        if(fp == 0)
        {
                printf("failed to open file");
                return NULL;
        }

        // get the length of the source code
        fseek(fp, 0, SEEK_END);
        source_length = ftell(fp);
        rewind(fp);

        // allocate a buffer for the source code string and read it in
        source_string = (char *)malloc( source_length + 1);
        if( fread( source_string, 1, source_length, fp) !=source_length )
        {
                printf("\n\t Error : Fail to read file ");
                return 0;
        }

        source_string[source_length+1]='\0';

        fclose(fp);

        return source_string;

}// End of the fuction to read the 'kernel source 


inline void checkStatus(const char *name,cl_int err)
{
        if (err != CL_SUCCESS)
        {

                printf("\n\t\t Error: %s (%d) \n",name, err);
                exit(-1);
        }
}


/*****************************************************************************************
*Function for Initializing the array with Random values
*****************************************************************************************/
void fillArray (float * matrix, int rows, int cols )
{

        int     row_count, col_count ;

        for( row_count=0; row_count < rows; row_count++)
             for( col_count=0; col_count < cols; col_count++)
                        matrix[row_count * cols + col_count] = rand()%10;

}
/*****************************************************************************************************
Function to set the execution environment for opencl kernel 
*****************************************************************************************************/
void setExeEnv ( cl_context *context, cl_uint *num_devices, cl_device_id **devices, cl_program *program)
{
 
	cl_platform_id	platform_id; /* Hold the OpenCL Platform Id */
	cl_int 		err; /* hold the err sode */
	int		count; /* loop control variable */
	size_t	 	kernel_src_length; /* hold the kernel source string length */
	char 		*kernel_src_str; /* hold the kernel source string */			
	

	/** 5. Get the OpenCL Platform ID ****/
	if ( getPlatform(&platform_id)) {
                printf(" \n\t\t Failed to get platform Info \n");
                exit(-1);
        }


	/** 6. Get the count & list of available OpenCL devices  */
        err = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 0, 0, num_devices);
        if( err != CL_SUCCESS  || *num_devices == 0) {
                 printf("\n\t\t ERROR : Failed to get Device Ids Or No OpenCL device found  \n");
                 exit(-1);
         }
         else {
                   assert( ((*devices) = (cl_device_id *) malloc( sizeof(cl_device_id ) * (*num_devices))) != NULL);
                   err = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, (*num_devices), *devices, 0);
                   checkStatus ("clGetDeviceIDs Failed", err);
         }
		
	printf(" Devices found : %d \n", *num_devices);

	printf("\n OpenCL device info :");
	printf("\n _____________________________________________________________________________________\n");
	 for( count= 0; count < *num_devices; count++) {
		printf (" \n %d) Device Info \n ", count +1);
		if( getDeviceInfo ( (*devices)[count]) )
			printf("\n\t Failed to print the device info \n");
	}

	
	/****** 7. Create Context for GPU Devices ***********/
        *context = clCreateContext( NULL, *num_devices, *devices, 0, 0, &err);
         if ( err != CL_SUCCESS || context == 0)
        {
                printf("\n\t No GPU detected ");
                printf("\n\t Context : %d , Err : %d",context, err);
                exit(-1);
        }

	 /******* 8. Create Program object from the kernel source  *****************/
	kernel_src_str = readKernelSource(kernel_source_path);
        kernel_src_length = strlen(kernel_src_str);
        *program = clCreateProgramWithSource( *context, 1, (const char **)&kernel_src_str, &kernel_src_length, &err);
        checkStatus( " clCreateProgramWithSource Failed ",err);

         /******* 9. Build (compile & Link ) Program ***************/
        err = clBuildProgram( *program,  *num_devices , *devices, NULL, NULL, NULL);
        checkStatus( " Build Program Failed ",err);


}
/*****************************************************************************************
*Function to perform matrix matrix multiplication on device 
*****************************************************************************************/
void	matMatMultiplication (cl_uint num_devices,cl_device_id *devices, cl_program program,cl_context context,float * h_a, float *h_b, float *h_output)
{

        cl_command_queue        *cmd_queue;     // OpenCL Command Queue 
        cl_mem                  *d_a, *d_b;     // OpenCL device input buffer
        cl_mem                  *d_output;      // OpenCL device output buffer
        cl_kernel               *kernel;        // OpenCL kernel
        cl_int                  err;            // Holds the status of API call
        cl_event                *time;
	double			total_time=0.0;
       
	size_t                  global_work_size[2];    // ND Range Size
        size_t                  local_work_size[2];    // ND Range Size

        int                      partition_size; // Data partition Size
        int                      num_threads;
	int			 row_count, col_count,count;

	double	gflops;


        /*******  10. Setting the Number of threads **********/
        num_threads =  num_devices;
        omp_set_num_threads(num_threads);


        /*** 11.  Distributing the Rows of Input Matrix A
            between the number of devices 
         */
        if ( height_a  % num_devices != 0 )
        {
                printf("\n\t Error :  Number of rows should be perfectly divisible by the number of threads \n");
                exit(-1);
        }
        else
                partition_size = height_a / num_devices;
                        
	printf("\n Devices Used \t\t\t\t: %d  ", num_devices);
        printf("\n Partition size\t\t\t\t: %d rows/device ", partition_size);

 	/*** 12. allocating memory for device variables */
        assert((d_a = (cl_mem *) malloc (sizeof(cl_mem) * num_threads)) != NULL);
        assert((d_b = (cl_mem *) malloc (sizeof(cl_mem) * num_threads)) != NULL);
        assert((d_output = (cl_mem *) malloc (sizeof(cl_mem) * num_threads)) != NULL);
        assert((kernel = (cl_kernel *) malloc (sizeof(cl_kernel) * num_threads)) != NULL);
        assert((cmd_queue = (cl_command_queue *) malloc (sizeof(cl_command_queue) * num_devices)) != NULL);
        assert((time = (cl_event *) malloc (sizeof(cl_event) * num_devices)) != NULL);


         /*** 13. OpenMP : Create the team of threads */
        #pragma omp parallel private(err) 
        {
		if ( omp_get_thread_num() == 0)
			printf("\n Threads Used\t\t\t\t: %d \n ", omp_get_num_threads());

                /******** 14  . Create the command queue for requested devices *********/
                cmd_queue[omp_get_thread_num()] = clCreateCommandQueue( context, devices[omp_get_thread_num()], CL_QUEUE_PROFILING_ENABLE, &err);
                if( err != CL_SUCCESS || cmd_queue[omp_get_thread_num()] == 0)
                {
                        printf("\n\t Failed to create command queue  \n" );
                        exit (-1);
                }


                /******* 15. Allocate and initialize memory buffer on each device  ****************/
                d_a[omp_get_thread_num()] = clCreateBuffer ( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR ,
							     (width_a * partition_size)  * sizeof(float),
                                                             &h_a[omp_get_thread_num() * partition_size * width_a], &err);
                checkStatus("Failed to create device input buffer A  ",err);

                /** Allocate memory / Copy data for input buffer B on device **/
                d_b[omp_get_thread_num()] = clCreateBuffer ( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , 
							     (height_b * width_b) * sizeof(float),h_b, &err);
                checkStatus("Failed to create device input buffer B  ",err);

                /** Allocate memory for output buffer on device **/
                d_output[omp_get_thread_num()] = clCreateBuffer ( context, CL_MEM_WRITE_ONLY ,( width_b * partition_size ) * sizeof(float),
								  NULL, &err);
                checkStatus( "Failed to create device output  buffer   ",err);

                /** Allocate memory / Copy value for a variable to hold Num Cols **/
                cl_mem d_rows =  clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), (void*) &width_a, &err);
                checkStatus( "Failed to create device output  buffer   ",err);

                /** Allocate memory / Copy value for a variable to hold Num Cols **/
                cl_mem d_cols =  clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), (void*) &width_b, &err);
                checkStatus( "Failed to create device output  buffer   ",err);


                /******* 16. Create the kernel **************/
                kernel[omp_get_thread_num()] = clCreateKernel ( program, "matMatMultKernel", &err);
                checkStatus(" Create kernel failed ",err);

                /******* 17. Set the kernel arguments ********/
 		err |= clSetKernelArg( kernel[omp_get_thread_num()], 0, sizeof(cl_mem), (void *) &d_a[omp_get_thread_num()]);
                err |= clSetKernelArg( kernel[omp_get_thread_num()], 1, sizeof(cl_mem), (void *) &d_b[omp_get_thread_num()]);
                err |= clSetKernelArg( kernel[omp_get_thread_num()], 2, sizeof(cl_mem), (void *) &d_output[omp_get_thread_num()]);
                err |= clSetKernelArg( kernel[omp_get_thread_num()], 3, sizeof(cl_mem), (void *) &d_rows);
                err |= clSetKernelArg( kernel[omp_get_thread_num()], 4, sizeof(cl_mem), (void *) &d_cols);
                checkStatus( "Set  kernel arguments failed ",err);


		/** 18. Setting the Global & Local work group size **/
                global_work_size [0]= width_b   ; // ND Range Size for each kernel launch 
                global_work_size[1]= partition_size ; // ND Range Size for each kernel launch

                /*local_work_size [0]= BLOCK_SIZE    ; // ND Range Size for each kernel launch 
                local_work_size[1]= BLOCK_SIZE   ; // ND Range Size for each kernel launch
		*/

                /****** 19. Enqueue / launch the kernel *******/
                //err = clEnqueueNDRangeKernel( cmd_queue[omp_get_thread_num()], kernel[omp_get_thread_num()], 2, NULL, global_work_size , 
		//			      local_work_size  , 0, NULL,&time[omp_get_thread_num()] );
                err = clEnqueueNDRangeKernel( cmd_queue[omp_get_thread_num()], kernel[omp_get_thread_num()], 2, NULL, global_work_size , 
					     NULL   , 0, NULL,&time[omp_get_thread_num()] );
                checkStatus( "  Kernel launch failed ",err);
		clFinish(cmd_queue[omp_get_thread_num()]);

                /****** 20. Read Results from the device ******/
                err = clEnqueueReadBuffer( cmd_queue[omp_get_thread_num()], d_output[omp_get_thread_num()], CL_TRUE, 0, 
					   width_b * partition_size * sizeof(cl_float), 
                                           &h_output[omp_get_thread_num() * width_b * partition_size], 0, 0, 0 );
                checkStatus(" Read output failed ",err);

        }

	/** Print the kernel execution time **/
	 for( count= 0; count < num_devices; count++) {
		total_time += executionTime( time[count]);
                printf("Kernel execution time on GPU %d\t\t: %.5f sec\n", count, executionTime(time[count]));
        }



	// gflops= (1.0e-9  * ((2. * height_a  * width_a * width_b) / total_time));

	/*
	printf("\n\n Matrix Size \t Time (Sec)  \n");
        printf("--------------------------------------------\n");
        printf(" %d \t\t %.5lf \t %.5lf \n", matrixSize, (total_time/num_devices));
	*/

	/** 21. Cleanup **/
	for(count = 0; count < num_devices ; count++ )
    	{
            	if ( kernel[count] )	clReleaseKernel(kernel[count]);
        	if ( cmd_queue[count])	clReleaseCommandQueue(cmd_queue[count]);
		if ( time[count] )	clReleaseEvent(time[count]);
    	}

	if(d_a)		free(d_a);
	if(d_b)		free(d_b);
	if(d_output)	free(d_output);


}

/*****************************************************************************************
*Function to check the GPU results against CPU results
*****************************************************************************************/
int checkResult (float *h_a, float *h_b,float *output, int rows, int cols)
{

        int i,j,k,step=0;
        float *temp_out;
        float sum;
	float  errorNorm = 0.0;
        float  eps=EPS;
        float  relativeError=0.0;
        int     flag=0;


        assert(temp_out = (float *)malloc( sizeof(float) * rows * rows));


        for( i=0 ; i<rows ; i++)
        {
                for( j=0 ; j<rows  ; j++)
                {
                        sum = 0.00f;
                        for( k=0 ; k<cols  ; k++)
                        {
                                sum += h_a[i * cols   + k] * h_b[k * rows   + j];
                        }
                        temp_out[step++] = sum;
                }
        }

	for( i=0 ; i < rows  ; i++)
        {
                for( j=0 ; j < rows  ; j++)
                {
                       // if ( temp_out[i*rows +j] != output[i*rows +j])
                         //      return 1;
                        if (fabs(temp_out[i*rows +j]) > fabs(output[i*rows +j]))
                        relativeError = fabs((temp_out[i*rows +j] - output[i*rows +j]) / temp_out[i*rows +j]);
                        else
                        relativeError = fabs((output[i*rows +j] - temp_out[i*rows +j]) / output[i*rows +j]);

                        if (relativeError > eps && relativeError != 0.0e+00 ){
                                if(errorNorm < relativeError) {
                                errorNorm = relativeError;
                                flag=1;
                                }
                        }


                     // printf("\t %f ",temp_out[i*rows  +j ]);
                }
              //printf("\n");
        }
        if( flag == 1) {

                //printf(" \n Results verfication : Failed");
                printf(" \n Considered machine precision : %e", eps);
                printf(" \n Relative Error                  : %e", errorNorm);
                free(temp_out);
                return 1;

        }
        else {

                free(temp_out);
                return 0;
        }

}

/*******************************************
Function to print the input ouput matrices
********************************************/
void print( float *h_a, float *h_b, float *h_output )
{

	int row_count, col_count;

        
        printf("\n Input Matrix A \n");
        for(row_count=0; row_count < height_a; row_count++) {
              for(col_count=0; col_count < width_a; col_count++) {
                          printf(" \t %f ", h_a[row_count * width_a + col_count]) ;
                }
               printf("\n");
         }
         
         printf("\n Input Matrix B\n");
         for(row_count=0; row_count < height_b; row_count++)
         {
               for(col_count=0; col_count < width_b; col_count++)
               {
                      printf(" \t %f ", h_b[row_count * width_b + col_count]) ;
                }
               printf("\n");
         }



      printf("\n Output :\n");
        for(row_count=0; row_count < height_a; row_count++)
        {
             for(col_count=0; col_count < width_b; col_count++)
             {
                      printf(" \t %f ", h_output[row_count * width_b + col_count]) ;
             }
             printf("\n");
         }
        printf("\n");

}

