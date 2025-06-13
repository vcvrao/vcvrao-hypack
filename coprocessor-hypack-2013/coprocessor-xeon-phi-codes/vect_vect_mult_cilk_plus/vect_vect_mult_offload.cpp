/**********************************************************************
*                    C-DAC Tech Workshop : hyPACK-2013
	                    October 15-18,2013
 *Example 1   	: vect_vect_mult_offload.cpp
 *Objective     : Write An Cilk Plus Program For vector vector multiplication
 *Input         : 1)Vector size
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

__attribute__((target(mic))) int vector_size;

double *fill_vector( int vector_size);
__declspec(target(mic)) void vector_mult(double *A,double *B,double *C,int vector_size);


double walltime()
{
       double tsec=0.0;
       struct timeval mytime;
       gettimeofday(&mytime,0);
       tsec=(double)(mytime.tv_sec+mytime.tv_usec*1.0e-6);
       return tsec;
}

//allocates memory for an array of doubles , 
double *fill_vector( int vector_size)
{
	double *a;
        if(vector_size <= 0)
        {
                       cout<< "illegal range of vector_size" << vector_size << endl;
                       return 0;
	}
        a=new double[vector_size];

	//a=(double *)malloc(vector_size*vector_size*sizeof(double));
        if(!a)
        {
        	cout<< "memory allocation failed\n" << endl;
		return 0;
	}
		
	return a;
}

 __declspec(target(mic)) void vector_mult(double *A,double *B,double *C,int vector_size)
{
	
	_Cilk_for(int i=0;i<vector_size;i++)
	{
				
		C[i] += A[i] * B[i];
				
	}
	

}

int main(int argc,char * argv[])
{
	double *A,*B,*C;
        double *B_trans;
	vector_size=atoi(argv[1]);

	     if(argc!=3)
             {
             	cout<<"Insuffiecient arguments"<<endl;
	     }
	     double time_start,time_end;
	     double time_elapsed; 
	     __cilkrts_set_param("nworkers", argv[2]);
	     A = fill_vector(vector_size) ; // allocates     the memory for vector b                                                       
             if(!A)
             {
             	cout << "memory allocation is failed" << endl;
                return -1;
              }
	      B = fill_vector(vector_size) ; // allocates     the memory for vector b                                                       
	     if(!B)
	      {
	          cout << "memory allocation is failed" << endl;
	          return -1;
	      }
	     C = fill_vector(vector_size) ; // allocates     the memory for vector b                                                       
	     if(!C)
	     {
	           cout << "memory allocation is failed" << endl;
	           return -1;
	    }
	     B_trans = fill_vector(vector_size) ; 
	     if(!B_trans)
	    {
	       cout << "memory allocation is failed" << endl;
	    }
		A[0:vector_size]=5;

		B[0:vector_size]=4;
		C[0:vector_size]=0;
		/*for(int i=0;i<matrix_size;i++)
		{
			
			B_trans[i]=B[i];
			
		}
*/
		time_start=walltime();
		
		#pragma offload target(mic) in(vector_size) \
		in(B:length(vector_size)) \
		in(A:length(vector_size)) \
		inout(C:length(vector_size))
		{
			int current_dev = _Offload_get_device_number();
			printf("program runs on target device - %d", current_dev);
			vector_mult(A,B,C,vector_size);
		}

		time_end=walltime();
		time_elapsed=time_end-time_start;
		double gflops=(((((double)vector_size)*2))/time_elapsed);
		cout<<"For cilk plus:"<<endl;
		cout<<"_____________________________________________________________________"<<endl;
		cout<<"Size\t Time \t\t Gflops"<<endl;
		cout<<""<<vector_size<<"\t"<<time_elapsed<<"\t\t"<<gflops/1000000000<<endl;

		delete [] &(A[0]);
		delete [] &(B[0]);
		delete [] &(C[0]);
		delete [] &(B_trans[0]);

		return 0;
}
 
 
