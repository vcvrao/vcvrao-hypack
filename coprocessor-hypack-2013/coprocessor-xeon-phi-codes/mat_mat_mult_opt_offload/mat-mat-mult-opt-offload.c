/*******************************************************************************************************
 *                      CDAC Tech Workshop - hyPACK 2013
 *                             Oct 15 - 18 , 2013
 *
 * FILE         : mat-mat-mul-posix-memalign-optimised.c
 * 
 * INPUT        : Matrix size
 * 
 * OUTPUT       : Time Elapsed
 * 
 * CREATED      : October,2013
 * 
 * EMAIL        : hpcfte@cdac.in
 *              
 * ****************************************************************************************************/



#include<stdio.h>
#include<stdlib.h>
#include "offload.h"

#define ROWCHUNK 128
#define COLCHUNK 128

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
	int i,j,k,ii,jj,kk,size,rc;
	
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


	double start=dtime();
	#pragma offload target(mic) \
	in(Matrix_A:length(size*size)) \
	in(Matrix_B:length(size*size)) \
	inout(Matrix_C:length(size*size)) 
	{
	#pragma omp parallel for collapse(2) private(i,j,k) 
		for(i=0;i<size;i+=ROWCHUNK){
			for(j=0;j<size;j+=ROWCHUNK){
				for(k=0;k<size;k+=COLCHUNK){
			for(ii=i;ii<i+ROWCHUNK;ii+=6){
				for(kk=k;kk<k+COLCHUNK;kk++){
			#pragma ivdep
			#pragma vector aligned
			for(jj=j;jj<j+ROWCHUNK;jj++){
			Matrix_C[(ii*size)+jj]+=Matrix_A[(ii*size)+kk]*Matrix_B[kk*size+jj];
			Matrix_C[((ii+1)*size)+jj]+=Matrix_A[((ii+1)*size)+kk]*Matrix_B[kk*size+jj];
			Matrix_C[((ii+2)*size)+jj]+=Matrix_A[((ii+2)*size)+kk]*Matrix_B[kk*size+jj];
			Matrix_C[((ii+3)*size)+jj]+=Matrix_A[((ii+3)*size)+kk]*Matrix_B[kk*size+jj];
			Matrix_C[((ii+4)*size)+jj]+=Matrix_A[((ii+4)*size)+kk]*Matrix_B[kk*size+jj];
			Matrix_C[((ii+5)*size)+jj]+=Matrix_A[((ii+5)*size)+kk]*Matrix_B[kk*size+jj];
			}
				}
			}
				}
			}
		}
	}
	double end=dtime();
	double TimeElapsed=end-start;
	if(TimeElapsed > 0.0)
	printf("Time Elapsed = %lf\n",TimeElapsed);
	return 0;
}
