/**********************************************************************
 *                    C-DAC Tech Workshop : hyPACK-2013
 *                          October 15-18,2013
 *Example       : matmat_mult_native_2D.cpp
 *Objective     : Write An Cilk Plus Program For matrix matrix multiplication
 *Input         : 1)Matrix size
 *                2)Number of threads
 *Output        : 1)Time taken in seconds
 *                2)Gflops/sec
 *Created       : August-2013    
 *E-Mail        : Betatest@Cdac.In                                          
 **************************************************************************/


#include<iostream>
#include<errno.h>
#include<ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <assert.h>
#include <mkl.h>
#include<cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <sys/time.h>


using namespace std;

int matrix_size;
double **fill_matrix( int matrix_size);

void matrix_mult(double **A,\ 
		 double **B,\
		 double **C,int matrix_size);
double walltime()
{
       double tsec=0.0;
       struct timeval mytime;
       gettimeofday(&mytime,0);
       tsec=(double)(mytime.tv_sec+mytime.tv_usec*1.0e-6);
       return tsec;
}

//allocates memory for an array of doubles , 
double **fill_matrix( int matrix_size)
{
	double **a;
        if(matrix_size <= 0)
        {
                       cout<< "illegal range of vector_size" << matrix_size << endl;
                       return 0;
	}
        a=(double **)malloc(matrix_size*sizeof(double *));
	for(int i=0;i<matrix_size;i++)
	{
		a[i]=(double *)malloc(matrix_size*sizeof(double));
	}
        if(!a)
        {
        	cout<< "memory allocation failed\n" << endl;
		return 0;
	}
		
	return a;
}

void matrix_mult(double **A,\
		 double **B,\
		 double **C,int matrix_size)
{
	
	_Cilk_for(int i=0;i<matrix_size;i++)
	{
		_Cilk_for(int j=0;j<matrix_size;j++)
		{
		
			for(int k=0;k<matrix_size;k++)
			{
				C[i][j]+=A[i][k]*B[k][j];
				//C[i*matrix_size+j] +=  A[i*matrix_size+k]*B[k*matrix_size+j];
				//C[i][j] = __sec_reduce_add(A[i][0:matrix_size] * B[0:matrix_size][j]);
			}
	
		 }
		
	}
	

}

int main(int argc,char * argv[])
{
	double **A,**B,**C;
        double **B_trans;
	matrix_size=atoi(argv[1]);

	     if(argc!=3)
             {
             	cout<<"Insuffiecient arguments"<<endl;
	     }
	     double time_start,time_end;
	     double time_elapsed; 
	     __cilkrts_set_param("nworkers", argv[2]);
	     A = fill_matrix(matrix_size) ; // allocates     the memory for vector b                                                       
             if(!A)
             {
             	cout << "memory allocation is failed" << endl;
                return -1;
              }
	      
	     B= fill_matrix(matrix_size) ; // allocates     the memory for vector b                                                       
	     if(!B)
	      {
	          cout << "memory allocation is failed" << endl;
	          return -1;
	      }
	     C= fill_matrix(matrix_size) ; // allocates     the memory for vector b                                                       
	     if(!C)
	     {
	           cout << "memory allocation is failed" << endl;
	           return -1;
	    }
	     B_trans= fill_matrix(matrix_size) ; 
	     if(!B_trans)
	    {
	       cout << "memory allocation is failed" << endl;
	    }
		A[0:matrix_size][0:matrix_size]=5;

		B[0:matrix_size][0:matrix_size]=4;
		C[0:matrix_size][0:matrix_size]=0;
		time_start=walltime();
		
			matrix_mult(A,B,C,matrix_size);

		time_end=walltime();
		time_elapsed=time_end-time_start;
		double gflops1=(((((double)matrix_size*matrix_size*matrix_size)*2))/time_elapsed);
		cout<<"For cilk plus:"<<endl;
		cout<<"_____________________________________________________________________"<<endl;
		cout<<"Size\t Time \t\t Gflops"<<endl;
		cout<<""<<matrix_size<<"\t"<<time_elapsed<<"\t\t"<<gflops1/1000000000<<endl;

		for(int i=0;i<matrix_size;i++)
		{
			free(A[i]);
		}
		free(A);
		for(int i=0;i<matrix_size;i++)
		{
			free(B[i]);
		}
		free(B);
		for(int i=0;i<matrix_size;i++)
		{
			free(C[i]);
		}
		free(C);
		for(int i=0;i<matrix_size;i++)
		{
			free(B_trans[i]);
		}
		free(B_trans);

		return 0;
}
 
 
