
/********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

	Example   : TbbVecVecScalableMalloc.cpp
		    using TBB LIBRARY API and scalable_malloc memory.

	Demostrate: parrel_for().
		    blocked_ranged().   
		    scalable_malloc().
		    scalable_free()	

	Input     : vector size.

	Output    : Product of vectors, Time in seconds.

        Created   : August-2013

        E-mail    : hpcfte@cdac.in     

*********************************************************************/

/*System related header files*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <iostream>

/* TBB related header files*/
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/scalable_allocator.h>

using namespace tbb;
using namespace std;

/* Global variable declaration*/
size_t vecSize; 
float *vA, *vB, *vC;
size_t mapsize;

/*Function to filling a vector*/
void
fill_vector (float *buf)
{
	for (size_t i = 0; i < vecSize; i++){
		buf[i]=2.00f;
	}
}

/*Function to print vector*/
void
print_vector(float *buf)
{
	for (size_t i = 0; i < vecSize; i++){
		printf ("%lf ", buf[i]);
	}
	printf ("\n");
}

/*Function to map a vector using scalable malloc*/
float *
map_vector (float *vector, int vecSize)
{
	size_t mapsize;

	mapsize = vecSize * sizeof (float);
	vector = (float *) scalable_malloc(mapsize);	
	return vector;
}

/* Function to freeing memory using scalable free*/
void free_mem(float *vector)
{
	scalable_free (vector);
}

/* Structure for vector vector multiplication using TBB*/
struct ParVectorMult
{
	size_t vecSize;
	float *vC, *vA, *vB;
	void operator () (const blocked_range < size_t > &r) const{
	int i;
    	for (i = r.begin (); i != r.end (); ++i){
		vC[i] = vA[i] * vB[i];
       	}
	}
};


/* Vector vector multiplication using TBB*/
void
par_vector_vector_multiply ()
{
	float sum = 0.00;
	ParVectorMult pmat;
	pmat.vecSize = vecSize;
	pmat.vA = vA;
	pmat.vB = vB;
	pmat.vC = vC;
	parallel_for (tbb::blocked_range<size_t> (0, vecSize, 1000), pmat);
	for ( int i = 0; i < vecSize; ++i)
		sum += vC[i];
		vC[0] = sum;
}

/* main function*/
int
main (int argc, char **argv)
{

	printf("\n\t\t---------------------------------------------------------------------------");
	printf("\n\t\t Centre for Development of Advanced Computing (C-DAC):  March-2009");
	printf("\n\t\t Email : betatest@cdac.in");
	printf("\n\t\t---------------------------------------------------------------------------");

	if (argc != 2){
		printf ("\n Usage: executable # vector size \n");
		exit (-1);
	}

	/* User input taking*/	
	vecSize = atoi (argv[1]);
	tbb::task_scheduler_init init; 

//	mapsize = vecSize * sizeof (float);

	/*mappping vectors*/
	vA = map_vector (vA, vecSize);
	vB = map_vector (vB, vecSize);
	vC = map_vector (vC, vecSize);
  
	/* filling vectors*/
	fill_vector (vA);
	fill_vector (vB);
	printf ("\n\t\t Vector Filled ..!!");

//	print_vector(vA);
//	print_vector(vB);

	/*Vector vector multiplication */
	tick_count t0 = tick_count::now();
	par_vector_vector_multiply ();
	tick_count t1 = tick_count::now();
	
	/*Results*/
	printf("\n\t\t Vector Multiplication Done ..!!");
	printf("\n\t\t Result: %lf", vC[0]);
	printf("\n\t\t Vector Size: %ld",vecSize);
	printf("\n\t\t Time taken to perform: %lf Sec",(t1-t0).seconds() );
	
	/*Freeing memory*/
	free_mem (vA);
	free_mem (vB);
	free_mem (vC);
	printf("\n\t\t Memory freed up successfully ..!!\n\n");
}
