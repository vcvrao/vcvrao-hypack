/***************************************************************************************************
* 			CDAC Tech Workshop - 2013
* 			  Oct 15 - 18 , 2013
*
* FILE		: mat-mat-mul-mmap-chunk-openmp.c
*
* AUTHOR	: K V SRINATH
*
* INPUT		: Matrix Size
*
* OUTPUT	: Time Elapsed to compute matrix multiplication using mmap
*
* CREATED	: September,2013
*
* EMAIL		: srinathkv@cdac.in 	hpcfte@cdac.in
*
* LIMITATION	: Works only for Matrix size which are multiples of 1024 (1024x1024,2048x2048,...)
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
#define PAGE_SIZE 4096
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

			val=rand()/(RAND_MAX+1.0);
			//val=2.0;
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

	for(int i=0;i<size;i++)
	{
		for(int j=0;j<size;j++)
		{
			sum=0.0;
			for(int k=0;k<size;k++)
			{
				sum+=ma[(i*size)+k]*mb[(k*size)+j];
			}
			mr[(i*size)+j]=sum;
		}
	}
}

double nrmsdError(int size,\
                  double (*  M1), \
                  double (*  M2))
{
	double sum = 0.0;
	int i,j;
	double diff;
	
	for ( i = 0; i < size; ++i){
		for ( j = 0; j < size;++j) {
			diff = (M1[i*size+j]- M2[i*size+j]);
          sum += diff*diff;
        }
      }
 
	//printf("sum =%lf\n",sum);
      return(sqrt(sum/(size*size)));
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
	int size,i,j,w,k;
	double *mapA,*mapB,*mapR,*mapCheck;
	struct timeval start,end;
	int block=512;
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
	fill_zeroes("./data/checkMatrix",size);
	
	transpose("./data/Matrix_B","./data/Matrix_Bt",size);
	
	double sum=0.0;
	int offset=(size*sizeof(double)/PAGE_SIZE)*PAGE_SIZE;	
	
	int fa=open("./data/Matrix_A",O_RDONLY);
	int fb=open("./data/Matrix_B",O_RDONLY);
	int fbt=open("./data/Matrix_Bt",O_RDONLY);
	int fr=open("./data/Matrix_R",O_RDWR);
	int fc=open("./data/checkMatrix",O_RDWR);
	
	
	//------------------------ Multiplication starts -------------------- /

	gettimeofday(&start,NULL);
	for( i=0;i<size;i++)  // size = No of rows
	{
		mapA=(double *)mapFile(fa,size*sizeof(double),i*offset,0);
		mapR=(double *)mapFile(fr,size*sizeof(double),i*offset,1);
		for( j=0;j<size/block;j++)
		{
			mapB=(double *)mapFile(fbt,block*size*sizeof(double),j*block*offset,0);
			#pragma omp parallel for private(w,sum) 
			for(k=0;k<block;k++)
			{
				sum=0.0;
				for( w=0;w<size;w++)
				{
					sum+=mapA[w]*(mapB+k*size)[w];
				}
				#pragma omp critical
				mapR[j*block+k]=sum;
			}

			munmap(mapB,size*block*sizeof(double));
		}
		munmap(mapA,size*sizeof(double));
		munmap(mapR,size*sizeof(double));
	}
	gettimeofday(&end,NULL);

	//------------------------ Multiplication ends --------------- /
	
	
	double timeElapsed=time_elapsed(start,end);
	printf("Time Elapsed using chunk method       = %lf\n",timeElapsed);
	
	//print_matrix("./data/Matrix_R",size);
	
	//------------------------- ERROR CHECKING ------------------- /
	
	/*mapA=(double *)mapFile(fa,size*size*sizeof(double),0,0);
	mapB=(double *)mapFile(fb,size*size*sizeof(double),0,0);
	mapCheck=(double *)mapFile(fc,size*size*sizeof(double),0,1);
	mapR=(double *)mapFile(fr,size*size*sizeof(double),0,1);
	
	gettimeofday(&start,NULL);
	doMult(mapA,mapB,mapCheck,size);
	gettimeofday(&end,NULL);
	
	timeElapsed=time_elapsed(start,end);
	printf("Time Elapsed using sequential method  = %lf\n",timeElapsed);
	
	double error=nrmsdError(size,mapR,mapCheck);
	printf("RMS ERROR = %lf\n",error);
	
	munmap(mapA,size*size*sizeof(double));
	munmap(mapB,size*size*sizeof(double));
	munmap(mapR,size*size*sizeof(double));
	munmap(mapCheck,size*size*sizeof(double));
*/
	close(fa);
	close(fb);
	close(fbt);
	close(fr);
	close(fc);
	return 0;
}
