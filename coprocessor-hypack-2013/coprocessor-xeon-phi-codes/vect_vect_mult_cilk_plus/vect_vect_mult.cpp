/**********************************************************************
 *                    C-DAC Tech Workshop : hyPACK-2013
 *                          October 15-18,2013
 *Example 1     : vect_vect_mult.cpp
 *Objective     : Write An Cilk Plus Program For Vector Vector multiplication
 *Input         : 1)Vector size
 *                2)Number of threads
 *Output        : 1)Time taken in seconds
 *                2)Gflops/sec
 *Created       : August-2013    
 *E-Mail        : hypack2013@cdac.in                                          
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

int vector_size;
double *fill_vector( int vector_size);

void vector_mult(double *vector_a,\ 
		 double *vector_b,\
		 double *vector_c,int vector_size);
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
        a=(double *)malloc(vector_size*sizeof(double));
        if(!a)
        {
        	cout<< "memory allocation failed\n" << endl;
		return 0;
	}
		
	return a;
}

void vector_mult(double *vector_a,\
		 double *vector_b,\
		 double *vector_c,int vector_size)
{
	double sum=0.0;
	//vector_c[0:vector_size] = vector_a[0:vector_size] * vector_b[0:vector_size] ;
	_Cilk_for (int  i = 0 ; i < vector_size ; i++)
        {
	    		vector_c[i] +=  vector_a[i] * vector_b[i];
	}
}
	
int main(int argc,char * argv[])
{
	double *vector_a,*vector_b,*vector_c;
        double *B_trans;
	vector_size=atoi(argv[1]);

	     if(argc!=3)
             {
             	cout<<"Insuffiecient arguments"<<endl;
	     }
	     double time_start,time_end;
	     double time_elapsed; 
	     
	     __cilkrts_set_param("nworkers", argv[2]);
	     
	     vector_a = fill_vector(vector_size) ; // allocates     the memory for vector b                                                       
             if(!vector_a)
             {
             	cout << "memory allocation is failed" << endl;
                return -1;
              }
	      
	     vector_b = fill_vector(vector_size) ; // allocates     the memory for vector b                                                       
	     if(!vector_b)
	      {
	          cout << "memory allocation is failed" << endl;
	          return -1;
	      }
	      vector_c = fill_vector(vector_size) ; // allocates     the memory for vector b                                                       
	     if(!vector_c)
	     {
	           cout << "memory allocation is failed" << endl;
	           return -1;
	    }
	     B_trans= fill_vector(vector_size) ; 
	     if(!B_trans)
	    {
	       cout << "memory allocation is failed" << endl;
	    }
		vector_a[0:vector_size]=5;
		vector_b[0:vector_size]=4;
		vector_c[0:vector_size]=0;
		time_start=walltime();
		
			vector_mult(vector_a,vector_b,vector_c,vector_size);

		time_end=walltime();
		time_elapsed=time_end-time_start;

		double gflops1=(((((double)vector_size)*2))/time_elapsed);
		
		cout<<"For cilk plus:"<<endl;
		cout<<"_____________________________________________________________________"<<endl;
		cout<<"Size\t Time \t\t Gflops"<<endl;
		cout<<""<<vector_size<<"\t"<<time_elapsed<<"\t\t"<<gflops1/1000000000<<endl;

		free(vector_a);
		free(vector_b);
		free(vector_c);
		free(B_trans);

		return 0;
}
 
 
