/**********************************************************************
*                    C-DAC Tech Workshop : hyPACK-2013
	                    October 15-18,2013
 *Example 1   	: matrix_matrix_mult_offload.cpp
 *Objective     : Write An Cilk Plus Program For matrix matrix multiplication
 *Input         : 1)Matrix size
 		  2)Number of threads
 *Output        : 1)Time taken in seconds
 		  2)Gflops/sec
 *Created       : August-2013    
 *E-Mail      	: hypack2013@cdac.in                                          
 *************************************************************************/ 
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include <sys/time.h>
#include<errno.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include<ctype.h>
#pragma offload_attribute(push,target(mic))
#pragma offload_attribute(pop)

using namespace std;

__attribute__((target(mic))) int matrix_size;
//int matrix_size;
double *fill_matrix( int matrix_size);
__declspec(target(mic)) void matrix_mult(double *A,double *B,double *C,int matrix_size);

//void matrix_mult(double *A,double *B,double *C,int matrix_size);
double walltime()
{
       double tsec=0.0;
       struct timeval mytime;
       gettimeofday(&mytime,0);
       tsec=(double)(mytime.tv_sec+mytime.tv_usec*1.0e-6);
       return tsec;
}

//allocates memory for an array of doubles , 
double *fill_matrix( int matrix_size)
{
	double *a;
        if(matrix_size <= 0)
        {
                       cout<< "illegal range of vector_size" << matrix_size << endl;
                       return 0;
	}
        a=(double *)malloc(matrix_size*matrix_size*sizeof(double));
        if(!a)
        {
        	cout<< "memory allocation failed\n" << endl;
		return 0;
	}
		
	return a;
}

 __declspec(target(mic)) void matrix_mult(double *A,double *B,double *C,int matrix_size)
{
	
	_Cilk_for(int i=0;i<matrix_size;i++)
	{
		_Cilk_for(int j = 0 ; j < matrix_size ; j++)
		{
		
			for(int k = 0 ; k < matrix_size ; k++)
			{
				//C[i][j]+=A[i][k]*B[k][j];
				C[i*matrix_size+j] +=  A[i*matrix_size+k]*B[k*matrix_size+j];
				//C[i][j] = __sec_reduce_add(A[i][0:size] * B[0:size][j]);
			}
	
		// cout<<B[i][j];
		 }
		
	}
	

}

int main(int argc,char * argv[])
{
	double *A,*B,*C;
        double *B_trans;
	matrix_size=atoi(argv[1]);

	     if(argc!=3)
             {
             	cout<<"Insuffiecient arguments"<<endl;
	     }
	     double time_start,time_end;
	     double time_elapsed; 
	     __cilkrts_set_param("nworkers", argv[2]);
	     A = fill_matrix(matrix_size) ; // allocates the memory for matrix A                                                       
             if(!A)
             {
             	cout << "memory allocation is failed" << endl;
                return -1;
              }
	      B = fill_matrix(matrix_size) ; // allocates the memory for matrix b                                                       
	     if(!B)
	      {
	          cout << "memory allocation is failed" << endl;
	          return -1;
	      }
	     C = fill_matrix(matrix_size) ; // allocates the memory for matrix b                                                       
	     if(!C)
	     {
	           cout << "memory allocation is failed" << endl;
	           return -1;
	    }
	     B_trans = fill_matrix(matrix_size) ; 
	     if(!B_trans)
	    {
	       cout << "memory allocation is failed" << endl;
	    }
		A[0:matrix_size*matrix_size]=5;

		B[0:matrix_size*matrix_size]=4;
		C[0:matrix_size*matrix_size]=0;
		/*for(int i=0;i<matrix_size;i++)
		{
			
			B_trans[i]=B[i];
			
		}
*/
		time_start=walltime();
		
		#pragma offload target(mic) in(matrix_size) \
		in(B:length(matrix_size*matrix_size)) \
		in(A:length(matrix_size*matrix_size)) \
		inout(C:length(matrix_size*matrix_size))
		{
			int current_dev = _Offload_get_device_number();
			printf("program runs on target device - %d", current_dev);
			matrix_mult(A,B,C,matrix_size);
		}

		time_end=walltime();
		time_elapsed=time_end-time_start;
		double gflops1=(((((double)matrix_size*matrix_size*matrix_size)*2))/time_elapsed);
		cout<<"For cilk plus:"<<endl;
		cout<<"_____________________________________________________________________"<<endl;
		cout<<"Size\t Time \t\t Gflops"<<endl;
		cout<<""<<matrix_size<<"\t"<<time_elapsed<<"\t\t"<<gflops1/1000000000<<endl;

		free(A);
		free(B);
		free(C);
		free(B_trans);

		return 0;
}
 
 
