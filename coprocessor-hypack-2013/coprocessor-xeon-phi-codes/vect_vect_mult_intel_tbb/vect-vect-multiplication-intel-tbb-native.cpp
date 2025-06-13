////////////////////////////////////////////////////////////////////////////////////////////////////
//
//				CDAC Tech Workshop, hyPACK-2013
//					Oct 15-18, 2013
//
// File		: xeon-phi-tbb-vec-vec-add-native.cpp 
//
// Author	: K V SRINATH
//
// Input	: Vector size , no of threads
//
// Output	: Time Elapsed for vector addition using intel tbb library.
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
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/tick_count.h"

//#define DATA_TYPE float
#define DATA_TYPE double
using namespace std;
using namespace tbb;

class vectorMul 
{
	private:
		DATA_TYPE * pa;
		DATA_TYPE * pb;
		int size;
	public:
		DATA_TYPE *sum;
		void operator()( const blocked_range<size_t>& r )  
		{
	        	DATA_TYPE *a= pa;
			DATA_TYPE *b= pb;
			for( size_t i=r.begin(); i!=r.end(); ++i )
						sum[0]+= a[i]*b[i];
		}
		vectorMul(vectorMul& x,split):pa(x.pa),pb(x.pb),sum(x.sum),size(x.size){}
		vectorMul(DATA_TYPE *a,DATA_TYPE *b,DATA_TYPE *r,int sz):pa(a),pb(b),sum(r),size(sz){}
                void join(const vectorMul& y)
                {
                        sum[0]+=y.sum[0];
                }

};

template <class T>
void fill_vector(T *vecx,int size)
{
	for(size_t i=0;i<size;i++)
			vecx[i]=(DATA_TYPE)rand()/(RAND_MAX+1.0);
}
template <class T>
void print_vector(T *vecx,int size)
{
	for(size_t i=0;i<size;i++)
			cout<<vecx[i]<<"\t";

	cout<<"\n";
}
	
int main(int argc,char *argv[])
{
	DATA_TYPE *Vector_A,*Vector_B,*result;
	int nThreads,size;
	srand(time(0));
	if(argc<3)
	{
		cout<<"syntax <Vector size> <nThreads>\n";
		exit(1);
	}
	
	size=atoi(argv[1]);
	nThreads=atoi(argv[2]);

	//Dynamic memory allocation
	Vector_A=(DATA_TYPE *)malloc(size*sizeof(DATA_TYPE));
	if(Vector_A==NULL)
	{
		cout<<"malloc error\n";
		exit(1);
	}
	
	Vector_B=(DATA_TYPE *)malloc(size*sizeof(DATA_TYPE));
	if(Vector_B==NULL)
	{
		cout<<"malloc error\n";
		exit(1);
	}
	
	result=(DATA_TYPE *)malloc(sizeof(DATA_TYPE));
	

	fill_vector(Vector_A,size);
	fill_vector(Vector_B,size);
	result[0]=0.0f;
	vectorMul vec(Vector_A,Vector_B,result,size);

	cout<<"Vector A values are\n";
	print_vector(Vector_A,size);
	cout<<"Vector B values are\n";
	print_vector(Vector_B,size);


	task_scheduler_init init(nThreads);
	
	tick_count t0 = tick_count::now();
	parallel_reduce(blocked_range<size_t>(0,size,1),vec);

	tick_count t1 = tick_count::now();
	
	double timeOverhead=(t1-t0).seconds();
	printf("Parallel Time Elapsed %g seconds\n",(t1-t0).seconds());

	cout<<"MFLOPS="<<(((double)size/((t1-t0).seconds()))/1000000)<<endl;

	cout<<"Result ="<<result[0]<<endl;


	init.terminate();

	
	

	// free memory
	free(Vector_A);
	free(Vector_B);


}

