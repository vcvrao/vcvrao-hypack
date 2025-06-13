/***************************************************************************************************
* FILE		: mat-mat-mul-mmap-dgemm-native.c
*
* AUTHOR	: K V SRINATH
*
* INPUT		: Matrix Size
*
* OUTPUT	: Time Elapsed to compute matrix multiplication using mmap
*
* CREATED	: August,2013
*
* EMAIL		: srinathkv@cdac.in    hpcfte@cdac.in
*
***************************************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<sys/mman.h>
#include<sys/time.h>
#include<unistd.h>
#include<fcntl.h>
#include<mkl.h>
#include<offload.h>
int k=0;
double alpha=1.0;
double beta=0.0;
void fill_matrix(char *name,int size)
{
	int i,j;
	double val;
	FILE *fx=fopen(name,"wb");
	if(fx<0)
	{
		perror("file open error\n");
		exit(EXIT_FAILURE);
	}
	
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
		{

			//val=rand()/(RAND_MAX+1.0);
			val=1.0;
			fwrite(&val,sizeof(double),1,fx);
		}

	}
	fclose(fx);
}
void fill_zeroes(char *name,int size)
{
	int i,j;
	double val;
	FILE *fx=fopen(name,"wb");
	if(fx<0)
	{
		perror("file open error\n");
		exit(EXIT_FAILURE);
	}
	
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
		{

			val=0.0;
			fwrite(&val,sizeof(double),1,fx);
		}

	}
	fclose(fx);
}




void print_matrix(char *name,int size)
{

	double buf,val;
	int i,j;

	FILE *fx=fopen(name,"rb");
	if(fx<0)
	{
		perror("file open error\n");
		exit(EXIT_FAILURE);
	}
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
		{
			fread(&val,sizeof(double),1,fx);
			printf("%lf ",val);
		}
		printf("\n");
	}
	fclose(fx);
}

void transpose(char *name1,char *name2,int size)
{
	double val;
	int i,j;
	FILE *fx=fopen(name1,"rb");
	if(fx<0)
	{
		perror("file open error\n");
		exit(EXIT_FAILURE);
	}
	FILE *ft=fopen(name2,"wb");
	if(ft<0)
	{
		perror("file open error\n");
		exit(EXIT_FAILURE);
	}
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
		{	
			fseek(fx,sizeof(double)*(size*j+i),SEEK_SET);
			fread(&val,sizeof(double),1,fx);
			fwrite(&val,sizeof(double),1,ft);
		}
	}
	fclose(fx);
	fclose(ft);
}
		
void doMult(double *ma,double *mb,double *mr,int size)
{
	double sum=0.0;
	int i,j,k;
	//int micId=_Offload_get_device_number();
	//printf("MIC ID = %d\n",micId);
	#pragma omp parallel for private(i,j,k) reduction(+:sum)
	for( i=0;i<size;i++)
	{
		for( j=0;j<size;j++)
		{
			sum=0.0;
			for( k=0;k<size;k++)
			{
				sum+=ma[(i*size)+k]*mb[(k*size)+j];
			}
			mr[(i*size)+j]=sum;
		}
	}
}
void call_dgemm(double *restrict a,double * restrict b,double * restrict c ,int size)
{
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,size, size, size,alpha, a, size, b, size, beta, c, size);
	
}

double time_elapsed(struct timeval start,struct timeval end)
{
	double t_perf;
	if(end.tv_usec<start.tv_usec)
	{
		end.tv_usec=end.tv_usec+1000000;
		end.tv_sec=end.tv_sec-1;
	}
	end.tv_usec=end.tv_usec-start.tv_usec;
	end.tv_sec=end.tv_sec-start.tv_sec;
	t_perf=(double)end.tv_usec/1000000;
	t_perf=t_perf+(double)end.tv_sec;

	return t_perf;
}

void *mapFile(int fx,int length,int offset,int mode)
{

	void *mptr;
	if(mode==0)
	{
		mptr=mmap(NULL,length,PROT_READ,MAP_SHARED,fx,offset);
		if (mptr == MAP_FAILED)
		{
			close(fx);
			perror("Error mmapping the file");
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		mptr=mmap(NULL,length,PROT_READ | PROT_WRITE ,MAP_SHARED,fx,offset);
		if (mptr == MAP_FAILED)
		{
			close(fx);
			perror("Error mmapping the file");
			exit(EXIT_FAILURE);
		}
	}
	return mptr; 

}


int main(int argc,char *argv[])
{
	int size;
	double *mapA,*mapB,*mapR;
	struct timeval start,end;
	double timeElapsed;
	srand(time(NULL));
	if(argc<2)
	{
		printf("syntax <size>\n");
		exit(1);
	}
	size=atoi(argv[1]);

	fill_matrix("./data/Matrix_A",size);
	fill_matrix("./data/Matrix_B",size);
	fill_zeroes("./data/Matrix_R",size);
	
	//printf("Matrix A values are\n");
	//print_matrix("./data/Matrix_A",size);
	
	//printf("Matrix B values are\n");
	//print_matrix("./data/Matrix_B",size);
	
	//transpose("./data/Matrix_B","./data/Matrix_Bt",size);
	
	//printf("Matrix Bt values are\n");
	//print_matrix("./data/Matrix_Bt",size);

	int fa=open("./data/Matrix_A",O_RDONLY);
	int fb=open("./data/Matrix_B",O_RDONLY);
	int fbt=open("./data/Matrix_Bt",O_RDONLY);
	int fr=open("./data/Matrix_R",O_RDWR);
	
	
	
	mapA=(double *)mapFile(fa,size*size*sizeof(double),0,0);
	mapB=(double *)mapFile(fb,size*size*sizeof(double),0,0);
	mapR=(double *)mapFile(fr,size*size*sizeof(double),0,1);
	
	gettimeofday(&start,NULL);
	call_dgemm(mapA,mapB,mapR,size);
	gettimeofday(&end,NULL);
	
	timeElapsed=time_elapsed(start,end);
	printf("Time Elapsed using DGEMM      method  = %lf\n",timeElapsed);
	printf("GFLOPS       			      = %lf\n",((2.0*size*size*size)/(timeElapsed))/1E9);	
	
	munmap(mapA,size*size*sizeof(double));
	munmap(mapB,size*size*sizeof(double));
	munmap(mapR,size*size*sizeof(double));
	//print_matrix("./data/Matrix_R",size);

	close(fa);
	close(fb);
	close(fr);
	return 0;
}
