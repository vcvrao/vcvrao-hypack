/*****************************************************************************

   Created             : August-2013

   E-mail              : hpctfe@cdac.in     

	
		Stream UPC bench mark(MPI-UPC)
	rows,cols,SCALAR values can be altered by the user, 
        but rows must be a multiple of number of threads
	
Compiling:
		upcc -network=mpi -uses-mpi -link-with=mpicc stream.upc 
                 -I/usr/local/mpich2-1.0.7/include/ -lmpich 

Executing:
		upcrun -n 8 ./a.out 

Result:
		We get No of Operations per second for different Tests performed.

   Created             : August-2013

   E-mail              : hpctfe@cdac.in     


*******************************************************************************/

#include<upc.h>
#include<mpi.h>
#include<stdio.h>
#define rows 1000*THREADS 
#define cols 400
#define SCALAR 1021201.1212
shared [cols] double Loc1[rows*cols];
shared [cols] double Loc2[rows*cols];
int MYRANK;


/* Read from shared memory locations with affinity to reading thread */
void Local_Shared_Read(shared double* M)
{
	struct timeval start_time,end_time;
	gettimeofday(&start_time,NULL);
	double temp;
	upc_forall(int i=0;i<rows;i++;i%THREADS)
		for(int j=0;j<cols;j++)
			temp=M[i*rows+cols];
	gettimeofday(&end_time,NULL);
	int time=(end_time.tv_sec-start_time.tv_sec)*1000000+end_time.tv_usec-start_time.tv_usec;
	int sum;
	upc_barrier;
	MPI_Reduce(&time,&sum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	if(MYRANK==0)
		printf("Performance of \" Local Shared Read  \" is %e  operations per second \n",rows*cols*(double)(1000000)/(double)sum);
}



/* Write to shared memory locations with affinity to reading thread */
void Local_Shared_Set(shared double* M)
{
	struct timeval start_time,end_time;
	gettimeofday(&start_time,NULL);
	double temp;
	upc_forall(int i=0;i<rows;i++;i%THREADS)
		for(int j=0;j<cols;j++)
			M[i*rows+cols]=100.09;
	gettimeofday(&end_time,NULL);
	int time=(end_time.tv_sec-start_time.tv_sec)*1000000+end_time.tv_usec-start_time.tv_usec;
	int sum;
	upc_barrier;
	MPI_Reduce(&time,&sum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	if(MYRANK==0)
		printf("Performance of \"  Local Shared Set  \" is %e operations per second \n",rows*cols*(double)(1000000)/(double)sum);
}

/* copying from a shared memory location having affinity to reading thread, to another shared memory location having affinity to same thread */
void upc_stream_copy(shared double* M)
{
	struct timeval start_time,end_time;
	gettimeofday(&start_time,NULL);
	double temp;
	upc_forall(int i=0;i<rows;i++;i%THREADS)
		for(int j=0;j<cols;j++)
			Loc2[i*rows+cols]=M[i*rows+cols];
	gettimeofday(&end_time,NULL);
	int time=(end_time.tv_sec-start_time.tv_sec)*1000000+end_time.tv_usec-start_time.tv_usec;
	int sum;
	upc_barrier;
	MPI_Reduce(&time,&sum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	if(MYRANK==0)
		printf("Performance of \"  Local-Local copy  \" is %e operations per second \n",rows*cols*(double)(1000000)/(double)sum);
}


/* Multiplying contents of shared  memory location with scalar */
void upc_stream_scalar(shared double* M,double scalar)
{
	struct timeval start_time,end_time;
	gettimeofday(&start_time,NULL);
	double temp;
	upc_forall(int i=0;i<rows;i++;i%THREADS)
		for(int j=0;j<cols;j++)
			Loc2[i*rows+cols]=scalar*M[i*rows+cols];
	gettimeofday(&end_time,NULL);
	int time=(end_time.tv_sec-start_time.tv_sec)*1000000+end_time.tv_usec-start_time.tv_usec;
	int sum;
	upc_barrier;
	MPI_Reduce(&time,&sum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	if(MYRANK==0)
		printf("Performance of \"    scalar Mult     \" is %e operations per second \n",rows*cols*(double)(1000000)/(double)sum);
}


/* Adding contenets of one shared locations to another shared location */
void upc_stream_Add(shared double* M)
{
	struct timeval start_time,end_time;
	gettimeofday(&start_time,NULL);
	double temp;
	upc_forall(int i=0;i<rows;i++;i%THREADS)
		for(int j=0;j<cols;j++)
			M[i*rows+cols]+=Loc2[i*rows+cols];
	gettimeofday(&end_time,NULL);
	int time=(end_time.tv_sec-start_time.tv_sec)*1000000+end_time.tv_usec-start_time.tv_usec;
	int sum;
	upc_barrier;
	MPI_Reduce(&time,&sum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	if(MYRANK==0)
		printf("Performance of \"     addition       \" is %e operations per second \n",rows*cols*(double)(1000000)/(double)sum);
}


/* Adding the scalar multiple of shared memory location to another shared memory location */
void upc_stream_Triad(shared double* M,double scalar)
{
	struct timeval start_time,end_time;
	gettimeofday(&start_time,NULL);
	double temp;
	upc_forall(int i=0;i<rows;i++;i%THREADS)
		for(int j=0;j<cols;j++)
			M[i*rows+cols]+=Loc2[i*rows+cols]*scalar;
	gettimeofday(&end_time,NULL);
	int time=(end_time.tv_sec-start_time.tv_sec)*1000000+end_time.tv_usec-start_time.tv_usec;
	int sum;
	upc_barrier;
	MPI_Reduce(&time,&sum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	if(MYRANK==0)
		printf("Performance of \"      Triad         \" is %e  operations per second\n",rows*cols*(double)(1000000)/(double)sum);
}

int main()
{
	MPI_Comm_rank(MPI_COMM_WORLD,&MYRANK);
	upc_memset(Loc1,1020120.1212,sizeof(double)*rows*cols);
	upc_memset(Loc2,2123123.213123,sizeof(double)*rows*cols);
	Local_Shared_Set(Loc1);
	Local_Shared_Read(Loc1);
	upc_stream_copy(Loc1);
	upc_stream_scalar(Loc1,SCALAR);
	upc_stream_Add(Loc1);
	upc_stream_Triad(Loc1,SCALAR);
	if(MYRANK==0)printf("For Shared Memory with size %d x %d and block size %d..\n",rows,cols,cols);
	return 0;
}
