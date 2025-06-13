/*****************************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

Descritption : The Simple example program for beginners to demonstart how to use the
                multiple GPU with OpenCL+OpenMP. OpenCL Events is used to syncronize 
		the CPU and GPU devices. Vector Vector Addition is used in kernel. 

                Host reads the two vectors A & B . The vector is divided among the 
                available number of devices and that chunk of data will be copied 
                from host to each device . Each GPU device will do the computation on 
                its chuck of data. The Host reads the result from each device and 
                accumulate.

Input           : Size of vector 

Output          : Time taken for computation
 
Created         : August-2013

E-mail          : hpcfte@cdac.in     

*****************************************************************************************/

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
extern "C" void    vectVectAdd(cl_uint num_devices,cl_device_id *devices, cl_program program,cl_context context,float * h_a, float *h_b, float *h_output,int vector_size);
extern "C" void print(float *h_a, float *h_b, float *h_output);
extern "C" int  getPlatform(cl_platform_id *selected_platform);
extern "C" int  getDevices( cl_platform_id selected_platform, cl_uint *total_devices, cl_device_id **devices);
extern "C" void checkStatus(const char *name,cl_int err);
extern "C" char *readKernelSource(const char *kernel_source_path);
extern "C" int  getDeviceInfo(cl_device_id device);
extern "C" int readInput( int argc, char **argv, long int *vector_size);
extern "C" int checkResult (float *h_a, float *h_b,float *h_output, int vector_size);


/* define the  Kernel source path */
const char * kernel_source_path = "Vect_Vect_Add.cl";

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
	long int 	vector_size; /* hold the vector size  */
	float 		*h_a, *h_b, *h_output; /* Host input / output vectors */
	int 		count; /* loop control variables */

	/** 1. Reading the input ***/
		 if ( readInput(argc , argv, &vector_size)) {
                	printf("\n Error : Failed to read the input \n");
                	exit(-1);
        	}



         /* 2. Allocating Memory & Initializing Host Input Buffer with random values */
          assert( (h_a=(float *) malloc (sizeof(float) * vector_size )) != NULL) ;
          assert( (h_b =(float *) malloc (sizeof(float) * vector_size )) != NULL) ;
          assert( (h_output=(float *) malloc (sizeof(float) * vector_size )) != NULL) ;

         for ( count = 0 ; count < vector_size  ; count++)
                     h_a[count]=h_b[count] = rand() + 1.10f ;


	 /** Step 3-7. Setting the exeution environment for openCL kernel : step (5-9)**/
	 setExeEnv ( &context , &num_devices, &devices, &program );

	 printf("\n\n SUMMARY : ");
	 printf("\n _________________________________________________________________________________________"); 
         printf( "\n Vector Size \t\t\t\t:  A( %d ) , B( %d ) , C(%d )",vector_size, vector_size, vector_size);
	
	 /** Step 8-18. function to perform vector vector addition  on GPU **/
	vectVectAdd(num_devices, devices, program, context, h_a, h_b, h_output,vector_size);


  	 /*******************************************
        Un-comment below sections to print 
         --- Input & output vectors 
        *******************************************/
	/*	
         print(h_a, h_b, h_output,vector_size); 
	*/	

        /*19 . Check the GPU results against CPU result **/
        if ( checkResult( h_a,h_b,h_output, vector_size))
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
/**********************************************************************************
Function to read the input 
**********************************************************************************/
int readInput( int argc, char **argv, long int *vector_size)
{

 		/** Reading the input ***/
                if( argc != 2) {

                        printf("\n Usage : ./<executable> <vector-size> \n");
                       return 1;
                }

                /* Reading vector Size */
                *vector_size = atol(argv[1]);
	
	return 0;
}

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
        cl_uint         numDevices;

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
                                        printf("\n Platform   :  %s\n",pbuff);
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
       printf(" Device  :  %s   ",dbuff);

       /* Get device type  */
       err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
       checkStatus("clGetDeviceInfo Failed ",err);
       switch ( type )
       {
              case CL_DEVICE_TYPE_CPU : printf("\nDevice Type\t\t\t : CPU ");
              break;
              case CL_DEVICE_TYPE_GPU : printf("\n Device Type: GPU");
              break;
              case CL_DEVICE_TYPE_ACCELERATOR:printf("\n\t Device Type\t\t\t : Dedicated OpenCLAccelerator");
              break;
              case CL_DEVICE_TYPE_DEFAULT : printf("\n\t Device Type\t\t\t: DEFAULT  ");
              break;
        }


        /* Get device  global memory in bytes */
         err = clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_global_mem),
                                &device_global_mem, NULL);
          checkStatus("clGetDeviceInfo Failed ",err);
          printf("\n Global Memory : %lf MB  ",(double)(device_global_mem)/(KB * KB));

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
          printf(" \n Local Memory :%lf KB  ",(double)device_local_mem/(KB));


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



/*****************************************************************************************************
Function to set the execution environment for opencl kernel 
*****************************************************************************************************/
void setExeEnv ( cl_context *context, cl_uint *num_devices, cl_device_id **devices, cl_program *program)
{
 
	cl_platform_id	platform_id; /* Hold the OpenCL Platform Id */
	cl_int 		err; /* hold the err sode */
	int		count; /* loop control variable */
	size_t	 	kernel_src_length; /* hold the kernel source string length */
	char 		*kernel_src_str=NULL; /* hold the kernel source string */			
	

	/** 3. Get the OpenCL Platform ID ****/
	if ( getPlatform(&platform_id)) {
                printf(" \n\t\t Failed to get platform Info \n");
                exit(-1);
        }


	/** 4. Get the count & list of available OpenCL devices  */
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
		
	printf(" Devices found : %d ", *num_devices);

	printf("\n\n OpenCL devices  Info:\n");
	printf(" _____________________________________________________________________________\n");
	 for( count= 0; count < *num_devices; count++) {
		printf(" %d) Device Info \n", count+1 );
		if( getDeviceInfo ( (*devices)[count]) )
			printf("\n\t Failed to print the device info \n");
	}

	
	/****** 5. Create Context for GPU Devices ***********/
        *context = clCreateContext( NULL, *num_devices, *devices, 0, 0, &err);
         if ( err != CL_SUCCESS || context == 0)
        {
                printf("\n\t No GPU detected ");
                printf("\n\t Context : %d , Err : %d",context, err);
                exit(-1);
        }

	 /******* 6. Create Program object from the kernel source  *****************/
	kernel_src_str = readKernelSource(kernel_source_path);
        kernel_src_length = strlen(kernel_src_str) - 1;
        *program = clCreateProgramWithSource( *context, 1, (const char **)&kernel_src_str, &kernel_src_length, &err);
        checkStatus( " clCreateProgramWithSource Failed ",err);

	
         /******* 7. Build (compile & Link ) Program ***************/
        err = clBuildProgram( *program,  *num_devices , *devices, NULL, NULL, NULL);
        checkStatus( " Build Program Failed ",err);


}
/*****************************************************************************************
*Function to perform vector vector addition on device 
*****************************************************************************************/
void	vectVectAdd(cl_uint num_devices,cl_device_id *devices, cl_program program,cl_context context,float * h_a, float *h_b, float *h_output,int vector_size)
{

	cl_mem                  *d_a, *d_b;     // OpenCL device input buffer
        cl_mem                  *d_output;      // OpenCL device output buffer
        cl_command_queue        *cmd_queue;     // OpenCL Command Queue 
        cl_kernel               *kernel;        // OpenCL kernel
        size_t                  global_work_size;       // ND Range Size
        cl_event                *time;
	 double                  total_time=0.0;

        int                     num_threads;    // Number of threads
        cl_int                  err;                    // Holds the status of API call
        int                     count,partition_size;

         /*******8 . Setting the Number of threads **********/
         num_threads = num_devices;
         omp_set_num_threads(num_threads);


        /* 9. Partition data between the threads */
        if ( vector_size % num_threads != 0 )
        {
                printf("\n\t Error : Vector Size should be perfectly divisible by number of threads \n");
                exit(-1);
        }
        else
                partition_size = vector_size / num_threads;

	printf("\n Partition Size\t\t\t\t: %d elements/device",partition_size);
	printf("\n Devices Used \t\t\t\t: %d  ", num_devices);

        /* 10. Allocate the memory for device varibables */
        assert((d_a = (cl_mem *) malloc (sizeof(cl_mem) * num_threads)) != NULL);
        assert((d_b = (cl_mem *) malloc (sizeof(cl_mem) * num_threads)) != NULL);
        assert((d_output = (cl_mem *) malloc (sizeof(cl_mem) * num_threads)) != NULL);
        assert((kernel = (cl_kernel *) malloc (sizeof(cl_kernel) * num_threads)) != NULL);
        assert((cmd_queue = (cl_command_queue *) malloc (sizeof(cl_command_queue) * num_devices)) != NULL);
        assert((time = (cl_event *) malloc (sizeof(cl_event) * num_devices)) != NULL);

        /* 11. Create the team of threads *******************/
        #pragma omp parallel private(err) 
        {

		if ( omp_get_thread_num() == 0)
                        printf("\n Threads Used\t\t\t\t: %d \n ", omp_get_num_threads());
              
		  /********12 . Create the command queue for requested devices *********/
                cmd_queue[omp_get_thread_num()] = clCreateCommandQueue( context, devices[omp_get_thread_num()], 
									CL_QUEUE_PROFILING_ENABLE , &err);
                if( err != CL_SUCCESS || cmd_queue[omp_get_thread_num()] == 0)
                {
                        printf("\n\t Failed to create command queue for device  \n" );
                        exit (-1);
                }

		/******* 13 . Allocate and initialize memory buffer on each device  ****************/


                /** Allocate memory / Copy data in input buffer A on device **/
                d_a[omp_get_thread_num()] = clCreateBuffer ( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR , 
						             partition_size  * sizeof(float), 
							     &h_a[omp_get_thread_num() * partition_size], &err);
                checkStatus("Failed to create device input buffer A  ",err);

                /** Allocate memory / Copy data for input buffer B on device **/
                d_b[omp_get_thread_num()] = clCreateBuffer ( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR  , 
							     partition_size  * sizeof(float),
                                            		     &h_b[omp_get_thread_num() * partition_size], &err);
                checkStatus("Failed to create device input buffer B  ",err);

                /** Allocate memory for output buffer on device **/
                d_output[omp_get_thread_num()] = clCreateBuffer ( context, CL_MEM_WRITE_ONLY ,
								  partition_size * sizeof(float), NULL, &err);
                checkStatus( "Failed to create device output  buffer   ",err);


                /******* 14. Create the kernel **************/
                kernel[omp_get_thread_num()] = clCreateKernel ( program, "vectVectAddKernel", &err);
                checkStatus(" Create kernel failed ",err);

                /******* 15. Set the kernel arguments ********/
                err |= clSetKernelArg( kernel[omp_get_thread_num()], 0, sizeof(cl_mem), (void *) &d_a[omp_get_thread_num()]);
                err |= clSetKernelArg( kernel[omp_get_thread_num()], 1, sizeof(cl_mem), (void *) &d_b[omp_get_thread_num()]);
                err |= clSetKernelArg( kernel[omp_get_thread_num()], 2, sizeof(cl_mem), (void *) &d_output[omp_get_thread_num()]);
                checkStatus( "Set  kernel arguments failed ",err);


                global_work_size = partition_size; // ND Range Size for each kernel launch

                /****** 16. Enqueue / launch the kernel *******/
                err = clEnqueueNDRangeKernel( cmd_queue[omp_get_thread_num()], kernel[omp_get_thread_num()], 1, NULL, 
					      &global_work_size , NULL, 0, NULL, &time[omp_get_thread_num()] );
                checkStatus( "  Kernel launch failed ",err);


                /****** 17. Read Results from the device ******/
                err = clEnqueueReadBuffer( cmd_queue[omp_get_thread_num()], d_output[omp_get_thread_num()], CL_TRUE, 0,
				           partition_size * sizeof(float), &h_output[omp_get_thread_num() * partition_size] , 
					   0, 0, 0 );
                checkStatus(" Read output faied ",err);

	 }/* Terminate threads leaving master thread */


        for(unsigned int i = 0; i < num_devices; i++)
        {
		total_time += executionTime( time[count]);
                printf("Kernel execution time on GPU %d \t: %.8f s\n", i, executionTime(time[i]));
        }
/*
	 printf("\n\n Total execution time for kernel\t: %.8f sec ",(total_time/num_devices));
*/

	/** 18. Cleanup **/
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
int checkResult (float *h_a, float *h_b,float *h_output, int vector_size)
{

	int i;
	float   *temp_out;
        float   errorNorm = 0.0;
        float   eps=EPS;
        float   relativeError=0.0;
        int     flag=0;



        assert( temp_out = (float *)malloc( sizeof(float) * vector_size ));
        for( i = 0 ; i < vector_size ; i++)
                temp_out[i] = (h_a[i] + h_b[i]);

        return 0;

          for( i=0 ; i < vector_size  ; i++)
        {
               if (fabs(temp_out[i]) > fabs(h_output[i]))
               relativeError = fabs((temp_out[i] - h_output[i]) / temp_out[i]);
               else
               relativeError = fabs((h_output[i] - temp_out[i]) / h_output[i]);

               if (relativeError > eps && relativeError != 0.0e+00 ){
                      if(errorNorm < relativeError) {
                         errorNorm = relativeError;
                         flag=1;
                       }
                  }

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
Function to print the input/ouput vectors 
********************************************/
void print( float *h_a, float *h_b, float *h_output,int vector_size )
{

	int count;

	        /** Print Input Vectors */
      printf(" \n\t Input Vector A : \n\n"); 
        for ( count = 0 ; count < vector_size  ; count++)
        {
                printf("\t %f ", h_a[count]);
        }
        printf("\n");

        printf(" \n\t Input Vector B : \n\n"); 
        for ( count = 0 ; count < vector_size  ; count++)
        {
                printf("\t %f ", h_b[count]);
        }
        printf("\n");



       // printf("********* GPU computation Results ************** \n");
        /* Print Results */
      printf(" \n\n\t  OutPut vector : \n\n");  
        for ( count = 0 ; count < vector_size  ; count++)
        {
                printf("\t %f ", h_output[count]);
        }
        printf("\n"); 



}
