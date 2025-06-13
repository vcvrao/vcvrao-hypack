/************************************************************************************************************************************
			 C-DAC Tech Workshop : hyPACK-2013
                             October 15-18,2012

 Example 1.1           : vector-vector-addition-openmp-mic-offload-aos.c

 Objective             : Perform Vector-Vector Addition using intel-mic offload via Array of Structures (AOS)and openmp parallel for  
 
 Input                 : Set the vector size and number of threads
		   
		   		For C shell use command :
	
					setenc MIC_ENV_PREFIX=MIC		
					setenv MIC_OMP_NUM_THREADS 4 
	
		   		For bash shell use command :
		
					export MIC_ENV_PREFIX=MIC
					export MIC_OMP_NUM_THREADS=4 
 
 Output                : Time taken for computing vector addition.	                                            
                                                                        
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

struct myvec
{
	double a;
	double b;
	double c;
};

int main(int argc,char *argv[] )
{
	int size, numthreads, i, j;
	double start,end,timeElapsed,k=2;
	if(argc<3)
	{
		printf("syntax <size> <numthreads>\n");
		exit(1);
	}

	size=atoi(argv[1]);
	numthreads=atoi(argv[2]);

	struct myvec * vec_ptr=(struct myvec *)_mm_malloc(size*sizeof(struct myvec),64);
	
	printf("Initialisation of vectors started\n");
	for(i=0;i<size;i++)
	{
		vec_ptr[i].a=1.0F;
		vec_ptr[i].b=2.0F;
		vec_ptr[i].c=0.0F;
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
	inout(vec_ptr:length(size))
	#pragma omp parallel for
	for(i=0;i<size;i++)
	{
		#pragma vector aligned
		vec_ptr[i].c=k*vec_ptr[i].a+vec_ptr[i].b;
	}

	end=wallTime();
	timeElapsed=end-start;
	printf("Time Elapsed = %lf\n",timeElapsed);
#endif

	printf("vec_ptr[0].c=%lf\n", vec_ptr[0].c);
	
	_mm_free(vec_ptr);

	return 0;
}



	
