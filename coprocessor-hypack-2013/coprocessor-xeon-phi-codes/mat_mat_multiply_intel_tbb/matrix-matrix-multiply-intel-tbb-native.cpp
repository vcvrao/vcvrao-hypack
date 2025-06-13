////////////////////////////////////////////////////////////////////////////////////////////////////
//
//				CDAC Tech Workshop, hyPACK-2013
//					Oct 15-18, 2013
//
// File		: xeon-phi-tbb-mat-mat-mul-native.cpp 
//
// Author	: K V SRINATH
//
// Input	: Matrix size , no of threads
//
// Output	: Time Elapsed for matrix multiplication using intel tbb library.
//
// Created	: May,2013
//
// Email	: srinathkv@cdac.in     hpcfte@cdac.in
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range2d.h"
#include "tbb/parallel_for.h"
#include "tbb/tick_count.h"

//#define DATA_TYPE float
#define DATA_TYPE double
using namespace std;
using namespace tbb;

class matrixMult 
{
	private:
		DATA_TYPE **pa;
		DATA_TYPE **pb;
		DATA_TYPE **pr;
		int size;
	public:
		void operator()( const blocked_range2d<size_t>& r )  const
		{
	        	DATA_TYPE **a= pa;
			DATA_TYPE **b= pb;
			DATA_TYPE **r1= pr;
			for( size_t i=r.rows().begin(); i!=r.rows().end(); ++i )
			{
				for( size_t j=r.cols().begin(); j!=r.cols().end(); ++j ) 
				{
					DATA_TYPE sum = (DATA_TYPE)0.0;
					for( size_t k=0; k<size; ++k )
						sum += a[i][k]*b[k][j];
					r1[i][j] = sum;
				}
			}
		}
		matrixMult(DATA_TYPE **a,DATA_TYPE **b,DATA_TYPE **r,int sz):pa(a),pb(b),pr(r),size(sz){}
};

class matrixMultSeq 
{
	private:
		DATA_TYPE **pa;
		DATA_TYPE **pb;
		DATA_TYPE **pr;
		int size;
	public:
		void doMultSeq()
		{
	        	DATA_TYPE **a= pa;
			DATA_TYPE **b= pb;
			DATA_TYPE **r1= pr;
			for( size_t i=0; i<size; ++i )
			{
				for( size_t j=0; j<size; ++j ) 
				{
					DATA_TYPE sum = (DATA_TYPE)0.0;
					for( size_t k=0; k<size; ++k )
						sum += a[i][k]*b[k][j];
					r1[i][j] = sum;
				}
			}
		}
		matrixMultSeq(DATA_TYPE **a,DATA_TYPE **b,DATA_TYPE **r,int sz):pa(a),pb(b),pr(r),size(sz){}
};
template <class T>
void fill_matrix(T **matx,int size)
{
	for(size_t i=0;i<size;i++)
		for(size_t j=0;j<size;j++)
			matx[i][j]=(DATA_TYPE)rand()/(RAND_MAX+1.0);
}
template <class T>
void print_matrix(T **matx,int size)
{
	for(size_t i=0;i<size;i++)
	{
		for(size_t j=0;j<size;j++)
			cout<<matx[i][j]<<"\t";
		cout<<"\n";
	}
}
	
int main(int argc,char *argv[])
{
	DATA_TYPE **Matrix_A,**Matrix_B,**Matrix_R,**Matrix_R1;
	int nThreads,size;
	FILE *fp,*fs;
	srand(time(0));
	if(argc<3)
	{
		cout<<"syntax <Matrix size> <nThreads>\n";
		exit(1);
	}
	
	size=atoi(argv[1]);
	nThreads=atoi(argv[2]);

	//Dynamic memory allocation
	Matrix_A=(DATA_TYPE **)malloc(size*sizeof(DATA_TYPE *));
	if(Matrix_A==NULL)
	{
		cout<<"malloc error\n";
		exit(1);
	}
	for(size_t i=0;i<size;i++)
	{
		Matrix_A[i]=(DATA_TYPE *)malloc(size*sizeof(DATA_TYPE ));
		if(Matrix_A[i]==NULL)
		{
			cout<<"malloc error\n";
			exit(1);
		}
	}
	Matrix_B=(DATA_TYPE **)malloc(size*sizeof(DATA_TYPE *));
	if(Matrix_B==NULL)
	{
		cout<<"malloc error\n";
		exit(1);
	}
	for(size_t i=0;i<size;i++)
	{
		Matrix_B[i]=(DATA_TYPE *)malloc(size*sizeof(DATA_TYPE ));
		if(Matrix_B[i]==NULL)
		{
			cout<<"malloc error\n";
			exit(1);
		}
	}
	Matrix_R=(DATA_TYPE **)malloc(size*sizeof(DATA_TYPE *));
	if(Matrix_R==NULL)
	{
		cout<<"malloc error\n";
		exit(1);
	}
	for(size_t i=0;i<size;i++)
	{
		Matrix_R[i]=(DATA_TYPE *)malloc(size*sizeof(DATA_TYPE ));
		if(Matrix_R[i]==NULL)
		{
			cout<<"malloc error\n";
			exit(1);
		}
	}
	Matrix_R1=(DATA_TYPE **)malloc(size*sizeof(DATA_TYPE *));
	if(Matrix_R1==NULL)
	{
		cout<<"malloc error\n";
		exit(1);
	}
	for(size_t i=0;i<size;i++)
	{
		Matrix_R1[i]=(DATA_TYPE *)malloc(size*sizeof(DATA_TYPE ));
		if(Matrix_R1[i]==NULL)
		{
			cout<<"malloc error\n";
			exit(1);
		}
	}

	fill_matrix(Matrix_A,size);
	fill_matrix(Matrix_B,size);


	//cout<<"Matrix A values are\n";
	//print_matrix(Matrix_A,size);
	//cout<<"Matrix B values are\n";
	//print_matrix(Matrix_B,size);


	task_scheduler_init init(nThreads);
	
	tick_count t0 = tick_count::now();
	static affinity_partitioner ap;
	parallel_for(blocked_range2d<size_t>(0,size,0,size),matrixMult(Matrix_A,Matrix_B,Matrix_R,size),ap);

	tick_count t1 = tick_count::now();
	
	double timeOverhead=(t1-t0).seconds();
	//printf("Parallel Time Elapsed %g seconds\n",(t1-t0).seconds());

	//cout<<"MFLOPS="<<(((double)size*size*size*2/((t1-t0).seconds()))/1000000)<<endl;

	//cout<<"Matrix R values are\n";
	//print_matrix(Matrix_R,size);


	init.terminate();

	matrixMultSeq sq(Matrix_A,Matrix_B,Matrix_R1,size);
	
	t0 = tick_count::now();
	sq.doMultSeq();
	t1 = tick_count::now();
	//printf("Sequential Time Elapsed %g seconds\n",(t1-t0).seconds());
	//cout<<"MFLOPS="<<(((double)size*size*size*2/((t1-t0).seconds()))/1000000)<<endl;

	
	
	fp=fopen("./par.dat","w+");
	fs=fopen("./seq.dat","w+");


	for(size_t i=0;i<size;i++)
	{
		for(size_t j=0;j<size;j++)
			fprintf(fp,"%lf ",Matrix_R[i][j]);
		fprintf(fp,"\n");
	}
	for(size_t i=0;i<size;i++)
	{
		for(size_t j=0;j<size;j++)
			fprintf(fs,"%lf ",Matrix_R1[i][j]);
		fprintf(fs,"\n");
	}

	cout<<"Matrix size="<<size<<"x"<<size<<"\t"<<"MFLOPS="<<((double)size*size*size*2/(timeOverhead)/1000000)<<"\t"<<"Time Elapsed="<<timeOverhead<<"\t"<<"NoOfThreads="<<nThreads<<endl;
	// free memory
	for(size_t i=0;i<size;i++)
	free(Matrix_A[i]);
	free(Matrix_A);
	for(size_t i=0;i<size;i++)
	free(Matrix_B[i]);
	free(Matrix_B);
	for(size_t i=0;i<size;i++)
	free(Matrix_R[i]);
	free(Matrix_R);
	for(size_t i=0;i<size;i++)
	free(Matrix_R1[i]);
	free(Matrix_R1);

	fclose(fp);
	fclose(fs);

}

