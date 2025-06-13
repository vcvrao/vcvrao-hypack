/*************************************************************************************************************************************
 					 C-DAC Tech Workshop : hyPACK-2013
 			                        October 15-18,2012

 Example 1.1           : mat-mat-addition-openmp-mic-offload-soa.c

 Objective             : Perform Matrix-Matrix Addition using intel-mic offload via Structure of Arrays 2D (SOA) and openmp parallel for  
 
 Input                 : Set the matrix size by changing "size" in compiler switch -DSIZE=size by editing makefile 
			 Set number of threads as commandline input
                   
                                For C shell use command :
               			
					setenv MIC_ENV_PREFIX=MIC
                                        setenv MIC_OMP_NUM_THREADS 4 
        
                                For bash shell use command :
                
					export MIC_ENV_PREFIX=MIC
                                        export MIC_OMP_NUM_THREADS=4 
 
 Output                : Time taken for computing matrix addition.                                                  
                                                                        
 Created               : August 2013 
       
 E-mail                : hpcfte@cdac.in    
************************************************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <offload.h>

double wallTime()
{
	double tsec=0.0;
	struct timeval mytime;
	gettimeofday(&mytime,0);
	tsec=(double)(mytime.tv_sec+mytime.tv_usec*1.0e-6);
	return tsec;
}

struct mymat
{
	double a2d[SIZE][SIZE];	
	double b2d[SIZE][SIZE];	
};


int main(int argc,char *argv[] )
{

	struct mymat * mat_ptr=(struct mymat *)_mm_malloc(sizeof(struct mymat)*1, 64);
	if(!mat_ptr)
	{
		printf("malloc error\n");
		exit(1);
	}

	//pointer to local 2d output array stored on stack which is a limited resource, could cause segmentation fault for large SIZE 
	//double c_2d[SIZE][SIZE];
	//double (* arr2d_ptr)[SIZE]=c_2d;
	
	//set memory on heap
	double (* arr2d_ptr)[SIZE]=(double (*)[SIZE])_mm_malloc((SIZE*SIZE)*sizeof(double), 64);
	
	if(!arr2d_ptr) 
	{
		printf("malloc error\n");
		exit(1);
	}
	

	int size, numthreads, i,j;
	double start,end,timeElapsed,k=2;
	if(argc!=2)
	{
		printf("syntax <numthreads>\n");
		printf("syntax <matrix size> given an compilation time with -DSIZE=size switch\n");
		exit(1);
	}

	numthreads=atoi(argv[1]);

	printf("Initialisation of matrices started\n");
	for(i=0;i<SIZE;i++)
	{
		for(j=0; j<SIZE; ++j)
		{
			mat_ptr->a2d[i][j]=1.0F;
			mat_ptr->b2d[i][j]=2.0F;
			arr2d_ptr[i][j]=0.0F;
		}
	}
/*
	#pragma offload target(mic:0)
	#pragma omp parallel
	#pragma omp master
	{

	#ifdef __MIC__
		printf("running on target\n");
	#else
		printf("running on host\n");
	#endif
	}
*/

#ifdef __INTEL_OFFLOAD
	printf("Started computing\n");

	omp_set_num_threads_target(DEFAULT_TARGET_TYPE, 0, numthreads);
	start=wallTime();	
	
	#pragma offload target(mic:0)\
	in(k)\
	inout(mat_ptr:length(1))\
	out(arr2d_ptr:length(SIZE))
	{
		#pragma omp parallel for
		for(int i=0; i<SIZE; ++i)
		{
			for(int j=0; j<SIZE; ++j)
			{
				arr2d_ptr[i][j]=k*mat_ptr->a2d[i][j]+mat_ptr->b2d[i][j];
			}
		}
	}
	end=wallTime();
	timeElapsed=end-start;
	printf("Time Elapsed = %lf\n",timeElapsed);
	
	for(i=0; i<SIZE; i++)
	{	
		for(j=0; j<SIZE; ++j)
		{
			//printf("%lf ", arr2d_ptr[i][j]);
		}
	}	
#endif

	_mm_free(mat_ptr);
	_mm_free(arr2d_ptr);

	return 0;
}



	
