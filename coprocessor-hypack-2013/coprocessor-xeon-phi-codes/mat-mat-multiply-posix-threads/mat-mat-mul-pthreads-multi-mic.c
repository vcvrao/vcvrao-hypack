/***************************************************************************************************
 *  				CDAC Workshop - hyPACK 2013
 *				    Oct 15 - 18 ,2013
 * 
 * FILE		: mat-mat-mul-pthreads-multi-mic.c
 *
 * INPUT	: #Matrix Size
 *
 * OUTPUT	: Time elapsed and GFLOPS
 *
 * DESCRIPTION	: To demonstrate how to run the code on multiple mics using Posix threads
 *
 * CREATED	: August,2013
 *
 * EMAIL	: hpcfte@cdac.in
 *
 **************************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>
#include <sys/time.h>
#include <offload.h>
#define NUM_DEV 2
struct matInfo
{
	double *a;
	double *b;
	double *c;
	int mic_id;
	int size;
};

double wallTime()
{
	double tsec=0.0;
	struct timeval mytime;
	gettimeofday(&mytime,0);
	tsec=(double)(mytime.tv_sec+mytime.tv_usec*1.0e-6);
	return tsec;
}

void fill_Matrix(double *x,int size)
{
	int i;
	for(i=0;i<size*size;i++)
		//x[i]=2.0;
		x[i]=rand()/(RAND_MAX+1.0);
}

void fill_zeroes(double *x,int size)
{
	int i;
	for(i=0;i<size*size;i++)
		x[i]=0.0;
}
void print_Matrix(double *x,int size)
{
	int i,j;
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
		{
			printf("%lf ",x[i*size+j]);
		}
		printf("\n");
	}
}

void *tbody(void *args)
{
	int i,j,k;
	int sig;
	struct matInfo *temp=(struct matInfo *)args;
	int tid=temp->mic_id;
	double *a=temp->a;
	double *b=temp->b;
	double *c=temp->c;
	long int size=temp->size;
	printf("In thread %d\n",tid);
	long int start=(tid*(size/NUM_DEV))*size;
	long int end = ((size*(tid+1))*size)/NUM_DEV;
	#pragma offload target(mic:tid) \
	in(a:length(size*size) )\
	in(b:length(size*size))\
	inout(c[start:end-start]) 
	{
		double sum=0.0;
	//	printf("\ntarget device = %d", _Offload_get_device_number());
	//	printf("tid = %d\n",tid);
	
        #pragma	omp parallel for private(i,j,k) reduction(+:sum) 
	for(i=tid*(size/NUM_DEV);i<(size*(tid+1))/NUM_DEV;i++)
	{
		for(j=0;j<size;j++)
		{
			sum=0.0;
			for(k=0;k<size;k++)
			{
				sum+=a[i*size+k]*b[k*size+j];
			}
			c[i*size+j]=sum;
		}
	}
	}
	pthread_exit(0);
	return 0;
}

void *(*ptb[2])(void * args)={tbody,tbody};

int main(int argc,char *argv[])
{
	pthread_t tid[NUM_DEV];
	pthread_attr_t attr[NUM_DEV];
	double *matA,*matB,*matR;
	double start,end;
	int size,i,rc;
 	struct timeval  tv_start,tv_end;
        struct timezone tz;
        double timeElapsed;
	struct matInfo mi[2];
	srand(time(NULL));
	if(argc<2)
	{
		perror("syntax <Matrix size>\n");
		exit(1);
	}
	size=atoi(argv[1]);
	//printf("size =%d\n",size);
	matA=(double *)malloc(sizeof(double)*size*size);
	if(matA==NULL)
	{
		perror("malloc error\n");
		exit(1);
	}
	
	matB=(double *)malloc(sizeof(double)*size*size);
	if(matB==NULL)
	{
		perror("malloc error\n");
		exit(1);
	}
	
	matR=(double *)malloc(sizeof(double)*size*size);
	if(matR==NULL)
	{
		perror("malloc error\n");
		exit(1);
	}


	fill_Matrix(matA,size);
	//printf("Matrix A values are\n");
	//print_Matrix(matA,size);
	
	fill_Matrix(matB,size);
	//printf("Matrix B values are\n");
	//print_Matrix(matB,size);
	
	fill_zeroes(matR,size);

	/* Initialise structure */
	for(i=0;i<NUM_DEV;i++)
	{
		mi[i].a=matA;
		mi[i].b=matB;
		mi[i].c=matR;
		mi[i].mic_id=i;
		mi[i].size=size;
	}
	pthread_attr_init(&attr[0]);
	pthread_attr_init(&attr[1]);
	pthread_attr_setdetachstate(&attr[0], PTHREAD_CREATE_JOINABLE);	
	pthread_attr_setdetachstate(&attr[1], PTHREAD_CREATE_JOINABLE);	
	
	start=wallTime();
	for( i=0;i<NUM_DEV;i++)
		pthread_create(&tid[i],&attr[i],ptb[i],&mi[i]);
	
  
	for( i=0;i<NUM_DEV;i++)
	{
		rc = pthread_join(tid[i],NULL);
		if(rc)
		{
			printf("\n Error : Failed to join threads \n");
			exit(0);
		}
	}
	end=wallTime();
	timeElapsed = end-start;

	printf("Time Elapsed = %lf\n",timeElapsed);
	printf("GFLOPS 	     = %lf\n",((2.0*size*size*size)/(timeElapsed))/1E9);
	pthread_attr_destroy(&attr[0]);
	pthread_attr_destroy(&attr[1]);
	//printf("Matrix R values are\n");
	//print_Matrix(matR,size);
	free(matA);
	free(matB);
	free(matR);

}



