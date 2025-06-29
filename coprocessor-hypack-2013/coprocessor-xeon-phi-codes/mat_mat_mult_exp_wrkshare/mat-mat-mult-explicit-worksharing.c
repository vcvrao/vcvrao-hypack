/*******************************************************************************************************
 *
 *                      CDAC Tech Workshop - hyPACK 2013
 *	                     Oct 15 - 18 , 2013
 * 
 * FILE         : mat-mat-mul-explicit-worksharing.c
 * 
 * INPUT        : Matrix size
 * 
 * OUTPUT       : Time Elapsed
 * 
 * CREATED      : October,2013
 * 
 * EMAIL        : hpcfte@cdac.in
 * 
 * ***************************************************************************************************/


#include<stdio.h>
#include<stdlib.h>
#include "offload.h"

double dtime()
{
	double tseconds = 0.0;
	struct timeval mytime;
	gettimeofday(&mytime,(struct timezone*)0);
	tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
	return( tseconds );
}

int main(int argc,char *argv[])
{
	double *Matrix_A,*Matrix_B,*Matrix_C;
	int i,j,k,size,rc;
	
	if(argc!=2)
	{
		printf("Syntax  <%s>  <Matrix size>\n",argv[0]);
		exit(1);
	}
	
	size=atoi(argv[1]);
	//printf("size=%d\n",size);

	rc=posix_memalign((void **)&Matrix_A,64,size*size*sizeof(double));
	rc=posix_memalign((void **)&Matrix_B,64,size*size*sizeof(double));
	rc=posix_memalign((void **)&Matrix_C,64,size*size*sizeof(double));

	//Initialisation of Matrices
	for(i=0;i<size*size;i++)
	{
		Matrix_A[i]=1.0;
		Matrix_B[i]=1.0;
		Matrix_C[i]=0.0;
	}


	double *off=Matrix_C;

	double start=dtime();
	#pragma omp parallel
	{
	#pragma omp sections
		{
		#pragma omp section        //on the target
			{
		{
		#pragma offload target(mic) \
		in(Matrix_A:length(size*size)) \
		in(Matrix_B:length(size*size)) \
		inout(off:length(size*size/2)) 
			{
		#pragma omp parallel for 
		for(i=0;i<size/2;i++){
			for(k=0;k<size;k++){
			#pragma vector aligned
			#pragma ivdep
				for(j=0;j<size;j++){
				off[i*size+j]+=Matrix_A[i*size+k]*Matrix_B[k*size+j];
				}
			}
		}
	}
			}
			}

		#pragma omp section          //on the host
			{
		for(i=size/2;i<size;i++){
			for(k=0;k<size;k++){
			#pragma vector aligned
			#pragma ivdep
				for(j=0;j<size;j++){
				Matrix_C[i*size+j]+=Matrix_A[i*size+k]*Matrix_B[k*size+j];
				}
			}
		}
			}
	}
		#pragma omp barrier
		
	}
	double end=dtime();
	double TimeElapsed=end-start;
	if(TimeElapsed > 0.0)
	printf("Time Elapsed = %lf\n",TimeElapsed);

/*	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
		{
			printf("%lf ",Matrix_C[i*size+j]);
		}
		printf("\n");
	}*/
	return 0;
}
