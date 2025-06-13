/*******************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

	mpi-upc vector-vector multiplication code.(MPI_UPC)
	Randomly Generates both the input vectors.
	VECT_SIZE must be a multiple of threads and can be varied by user
	BLK_SIZE(block size) must be <= VECT_SIZE/THREADS, 
        can be varied by user.

Compiling:
	  upcc -network=mpi -uses-mpi -link-with=mpicc vect_mul.upc 
           -I/usr/local/mpich2-1.0.7/include/ -lmpich 

Executing:
		upcrun -n 8 ./a.out

Result:
		Scalar result of vect-vect multiplication and time taken


   Created             : August-2013

   E-mail              : hpctfe@cdac.in     


*************************************************************************/

#include<stdio.h>
#include<mpi.h>
#include<upc.h>
#include<math.h>
#include<upc_relaxed.h>
#define VECT_SIZE 10000000*THREADS
#define BLK_SIZE  1000000
shared [BLK_SIZE] double v1[VECT_SIZE];
shared [BLK_SIZE] double v2[VECT_SIZE];
shared double result[THREADS];


int main()
{
	int start;
	if(MYTHREAD==0)printf("Enter any integer to start : ");
	scanf("%d",&start);
	srand(start);
	/* generate random vector input values */
	if(MYTHREAD==0)printf("Generating Random input vectors..\n");
	upc_forall(int i=0;i<VECT_SIZE;i++;&v1[i])
  	{  
		v1[i]=(double)rand();
		v2[i]=(double)rand();
  	}
	
	upc_barrier;
  	/* actual code starts from here */
  	int MYRANK,SIZE;
  	upc_barrier 121;
	
	/* get mpi rank */
  	MPI_Comm_rank(MPI_COMM_WORLD,&MYRANK);
  	MPI_Comm_size(MPI_COMM_WORLD,&SIZE);
	
	
	struct timeval t1,t2;
	
	/* Get start time of execution */
	gettimeofday(&t1,NULL);

	if(MYTHREAD==0)
		printf("Starting vector vector multiplication......\n"); 


  	result[MYTHREAD]=0.0;
	
	/* Calculate No of blocks for each Thread */
	int z=(VECT_SIZE/BLK_SIZE)/THREADS;
	if((VECT_SIZE/BLK_SIZE)%THREADS >= (MYTHREAD+1) )z++;
	
	/* Vect-vect multiplication */
	for(int m=0;m<z;m++)
	{
  		for(int j=0,i=(m*BLK_SIZE*THREADS)+MYTHREAD*BLK_SIZE;j<BLK_SIZE;i++,j++)
  		{
  			result[MYTHREAD]+=v1[i]*v2[i];
  		}
	}


  	double SUM;
  	double send=result[MYTHREAD];
  	upc_barrier 100;
	/* Add  the individual  results */
  	MPI_Reduce(&send,&SUM,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

  	gettimeofday(&t2,NULL);//get the sys time after execution...
  	int time=(t2.tv_sec-t1.tv_sec)*1000000+t2.tv_usec-t1.tv_usec;
  	if(MYRANK==0)
  		printf("result of vector-vector multiplication of size %d  is %e \nTime taken is %d microseconds.\n",VECT_SIZE,SUM,time);
  	return 0;
}

