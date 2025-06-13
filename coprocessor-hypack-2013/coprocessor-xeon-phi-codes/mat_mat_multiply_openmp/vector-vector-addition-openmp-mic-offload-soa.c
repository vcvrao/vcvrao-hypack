/*************************************************************************************************************************************
 					 C-DAC Tech Workshop : hyPACK-2013
 			                        October 15-18,2012

 Example 1.1           : vector-vector-addition-openmp-mic-offload-soa.c

 Objective             : Perform Vector-Vector Addition using intel-mic offload via Structure of Arrays (SOA) and openmp parallel for  
 
 Input                 : Set the vector size and number of threads
                   
                                For C shell use command :
               			
					setenv MIC_ENV_PREFIX=MIC
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
	double * m1;
	double * m2;
	double * m3;
};

int main(int argc,char *argv[] )
{
	struct myvec vec;
	struct myvec *vec_ptr=&vec;

	double *va,*vb,*vc;
	int size, numthreads, i,j;
	double start,end,timeElapsed,k=2;
	if(argc<2)
	{
		printf("syntax <size> <numthreads>\n");
		exit(1);
	}

	size=atoi(argv[1]);
	numthreads=atoi(argv[2]);

	va=(double *)_mm_malloc(size*sizeof(double),64);
	if(va==NULL)
	{
		perror("malloc error\n");
		exit(1);
	}
	
	vb=(double *)_mm_malloc(size*sizeof(double),64);
	if(vb==NULL)
	{
		perror("malloc error\n");
		exit(1);
	}
	vc=(double *)_mm_malloc(size*sizeof(double),64);
	if(vc==NULL)
	{
		perror("malloc error\n");
		exit(1);
	}
	
	printf("Initialisation of vectors started\n");
	for(i=0;i<size;i++)
	{
		va[i]=1.0F;
		vb[i]=2.0F;
		vc[i]=0.0F;
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
	in(va:length(size))\
	in(vb:length(size))\
	out(vc:length(size))\
	nocopy(vec)
	{
		vec.m1=va;
		vec.m2=vb;
		vec.m3=vc;

		#pragma omp parallel for
		for(i=0;i<size;i++)
		{
			#pragma vector aligned
			vec.m3[i]=k*vec.m1[i]+vec.m2[i];
		}
	}
	end=wallTime();
	timeElapsed=end-start;
	printf("Time Elapsed = %lf\n",timeElapsed);
	printf("vc[3]=%lf\n", vc[3]);
#endif

	_mm_free(va);
	_mm_free(vb);
	_mm_free(vc);

	return 0;
}



	
