/*************************************************************************************************************************************
 					 C-DAC Tech Workshop : hyPACK-2013
 			                        October 15-18,2012

 Example 1.1           : mat-mat-addition-openmp-mic-offload-soa.c

 Objective             : Perform Matrix-Matrix Addition using intel-mic offload via Structure of Arrays 2D (SOA) and openmp parallel for  
 
 Input                 : Set matrix size, number of threads as commandline input
                   
                                For C shell use command :
               			
					setenv MIC_ENV_PREFIX=MIC
                                        setenv MIC_OMP_NUM_THREADS 4 
        
                                For bash shell use command :
                
					export MIC_ENV_PREFIX=MIC
                                        export MIC_OMP_NUM_THREADS=4 
 
 Output                : Time taken for computing matrix matrix addition.                                                  
                                                                        
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

int main(int argc,char *argv[] )
{
	int size, numthreads, i,j;
	double start, end, timeElapsed, k=2;
	if(argc!=3)
	{
		printf("syntax <size> <numthreads>\n");
		exit(1);
	}

	size=atoi(argv[1]);
	numthreads=atoi(argv[2]);

	//set memory on heap
	double (* matA2d_ptr)[size]=(double (*)[size])_mm_malloc((size*size)*sizeof(double), 64);
	double (* matB2d_ptr)[size]=(double (*)[size])_mm_malloc((size*size)*sizeof(double), 64);
	double (* matC2d_ptr)[size]=(double (*)[size])_mm_malloc((size*size)*sizeof(double), 64);
	
	if(!matA2d_ptr || !matB2d_ptr || !matC2d_ptr) 
	{
		printf("malloc error\n");
		exit(1);
	}

	printf("Initialisation of matrices started\n");
	for(i=0;i<size;i++)
	{
		for(j=0; j<size; ++j)
		{
			matA2d_ptr[i][j]=1.0F;
			matB2d_ptr[i][j]=2.0F;
			matC2d_ptr[i][j]=0.0F;
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
	in(size)\
	in(matA2d_ptr:length(size*size))\
	in(matB2d_ptr:length(size*size))\
	out(matC2d_ptr:length(size*size))
	{
		struct myMat
		{
			double (* matA2d_ptr1)[size];
			double (* matB2d_ptr1)[size];
		};

		struct myMat m;
		m.matA2d_ptr1=matA2d_ptr;	
		m.matB2d_ptr1=matB2d_ptr;	

		#pragma omp parallel for
		for(int i=0; i<size; ++i)
		{
			for(int j=0; j<size; ++j)
			{
				matC2d_ptr[i][j]=k*m.matA2d_ptr1[i][j]+m.matB2d_ptr1[i][j];
			}
		}
	}

	end=wallTime();
	timeElapsed=end-start;
	printf("Time Elapsed = %lf\n",timeElapsed);
	
	for(i=0; i<size; i++)
	{	
		for(j=0; j<size; ++j)
		{
			//printf("%lf ", matC2d_ptr[i][j]);
		}
	}
	
#endif
	
	_mm_free(matA2d_ptr);
	_mm_free(matB2d_ptr);
	_mm_free(matC2d_ptr);

	return 0;
}



	
