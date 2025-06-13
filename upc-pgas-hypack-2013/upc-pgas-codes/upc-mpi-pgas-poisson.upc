
/*******************************************************************************	


		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013


        mpi-upc Program For Poisson 2-d Solver.
	TOPBOUNDARYVALUE,BOTTOMBOUNDARYVALUE,LEFTBOUNDARYVALUE,RIGHTBOUNDARYVALUE 
         are intial boundary values and can be changed by the user.

	rows,cols can be changed according to reqiured dimensions of matrix.
	MAX_ITERATIONS can be changed to limit the maximum dimensions.
	
Compiling:
		**** copy and paste the below command **** 
		upcc -network=mpi -uses-mpi -link-with=mpicc poisson.upc 
                   -I/usr/local/mpich2-1.0.7/include/ -lmpich 

Running:
	   upcrun -n 8 ./a.out
	    Here 8 is the no of Threads ,on which u want to launch the program.
            This can be changed by the user according to system specifications.


   Created             : August-2013

   E-mail              : hpctfe@cdac.in     

************************************************************************************/

#include<upc.h>
#include<stdio.h>
#include<mpi.h>
#include<sys/time.h>
#define TOLERANCE (1.0E-15)
#define TOPBOUNDARYVALUE  4.0f
#define BOTTOMBOUNDARYVALUE  3.0f
#define LEFTBOUNDARYVALUE  1.0f
#define RIGHTBOUNDARYVALUE  2.0f
#define rows (2*THREADS)	//Must be multiple of THREADS
#define cols (8)
#define MAX_ITERATIONS (2)
#define ABS(x) ((x)<0)?(-(x)):(x)

struct timeval t1,t2;
int MYRANK,SIZE;	//mpi variables
shared double *Matrix1,*Matrix2;
void Print(shared double*); /* Function to print the Result */

/* Function for Setting The Intial Boundary Values */
void Initialise(shared double* M)
{
	if(MYTHREAD==0)
	{
		for(int i=0;i<cols;i++)
			M[i]=TOPBOUNDARYVALUE;
	}
	if(MYTHREAD==THREADS-1)
	{
		for(int i=0;i<cols;i++)
			M[(rows-1)*cols+i]=BOTTOMBOUNDARYVALUE;
	}
	upc_forall(int i=0;i<rows;i++;i%THREADS)
	{
		M[i*cols]=LEFTBOUNDARYVALUE;
		M[i*cols+cols-1]=RIGHTBOUNDARYVALUE;
	}
}



/* Calculate New values for all the 2-d points except boundaries M1 contains Present values,M2 contains Updated values */
int Calculate(shared double* M1,shared double* M2)
{
	int error=0;int index;
	upc_forall(int i=1;i<(rows-1);i++;i%THREADS)//update all rows and col  except boundaries
	{
		for(int j=1;j<(cols-1);j++)//add the adjacent values in same row
		{
			index=i*cols+j;
			M2[index]=M1[index-1];//left val
			M2[index]+=M1[index+1];//right val
			M2[index]+=M1[index-cols];//top value
			M2[index]+=M1[index+cols];//bottom value
			M2[index]*=0.25;
			if((ABS(M2[index]-M1[index]))>TOLERANCE)//check the diff	
			{
				error++;
			}
		}	
	}
	upc_barrier;
	return error;
}

/* Printing the Result */
void Print(shared double* M)
{
	for(int i=0;i<rows;i++)
        {
                for(int j=0;j<cols;j++)
                        printf("%lf ",M[i*cols+j]);
                        printf("\n",MYTHREAD);
        }
	return;
}


/* Check whether approximate solution is obtained */
void Check(int iterations,int continues)
{
	printf("Computed in %d Iterations for size %d x %d with tolerance of %e\n",iterations,rows,cols,TOLERANCE);
	gettimeofday(&t2,NULL);
	int time=(t2.tv_sec-t1.tv_sec)*1000000+t2.tv_usec-t1.tv_usec;
	printf("Time taken = %d microseconds..\n",time);
       	if(iterations%2==0)
        {
      		Print(Matrix1);
                upc_global_exit(0);
        }
        else
        {
         	Print(Matrix2);
                upc_global_exit(0);
        }

}


int main()
{

	MPI_Comm_rank(MPI_COMM_WORLD,&MYRANK);
	MPI_Comm_size(MPI_COMM_WORLD,&SIZE);
	Matrix1=(shared double*)upc_all_alloc(rows*cols,sizeof(double));
	Matrix2=(shared double*)upc_all_alloc(rows*cols,sizeof(double));
	Initialise(Matrix1);
	Initialise(Matrix2);
	gettimeofday(&t1,NULL);
	upc_barrier ;
        int ITERATIONS=0;int continues;
        while(ITERATIONS <  MAX_ITERATIONS)
	{
		int k=Calculate(Matrix1,Matrix2);ITERATIONS++;
		upc_barrier;continues=0;
		MPI_Reduce(&k,&continues,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
		if(!continues&&MYTHREAD==0)Check(ITERATIONS,continues);
		k=Calculate(Matrix2,Matrix1);ITERATIONS++;continues=0;
		MPI_Reduce(&k,&continues,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
 		if(!continues&&MYTHREAD==0)Check(ITERATIONS,continues);
	}
		if(MYTHREAD==0)printf("Maximum iterations of  %d Completed \n",ITERATIONS);
	upc_barrier;
	if(MYTHREAD==0)
	Check(ITERATIONS,0);
	upc_barrier;
	return 0;
}

