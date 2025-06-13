
/**************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

	Example   : TbbMatVecScalableMalloc.cpp
		    using TBB LIBRARY API and scalable_malloc memory.

	Demostrate: parrel_for().
		    blocked_ranged().   
		    scalable_malloc().
		    scalable_free()	

	Input     : vector size.

	Output    : Product of matrix-vector, Time in seconds.

       Created    : August-2013
 
       E-mail    : hpcfte@cdac.in     

****************************************************************/

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
size_t vecSize, nRows, nColms; 
float **mA, *vB, *vR;
size_t mapsize;
size_t mapCsize;

/*Function to filling a vector*/
void
fill_vector (float *buf)
{
	for (size_t i = 0; i < vecSize; i++){
		buf[i] = 2.00f;
	}
}

/*Function to filling a matrix*/
void
fill_matrix (float **buf)
{
	for (size_t i = 0; i < nRows ; ++i){
	for (size_t j = 0; j < nColms; ++j){
		buf[i][j] = 2.00f;
	}
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

/*Function to print matrix*/
void
print_matrix(float **buf)
{
	for (size_t i = 0; i < nRows; i++){
	for (size_t j = 0; i < nColms; j++){
		printf ("%lf ", buf[i][j]);
	}
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

/*Function to map a matrix using scalable malloc*/
float **
map_matrix (float **matrix, int nRows, int nColms)
{
	size_t mapsize;
	size_t mapCsize;

	mapsize = nRows  * sizeof (float *);
	mapCsize = nColms  * sizeof (float);

	matrix = (float **) scalable_malloc(mapsize);
	for (int i = 0; i < nRows; i++)
		matrix[i] = (float *) scalable_malloc(mapCsize);

	return matrix;

}

/* Function to freeing memory using scalable free*/
void free_memV(float *vector)
{
	scalable_free (vector);
}
void free_memM(float **matrix)
{
	scalable_free (matrix);
}

/* Structure for vector vector multiplication using TBB*/
struct ParVectorMult
{
	size_t vecSize, nRows, nColms;
	float *vR,**mA, *vB;
	void operator () (const blocked_range < size_t > &r) const{
	int j, k ;
	float sum ;
    	for (j = r.begin (); j < r.end (); ++j){
		sum = 0.0;
	for (k = 0; k < nColms; ++k){
		sum += mA[j][k] * vB[k];
	}
		vR[j] = sum;
       	}
	}
	
};


/* Vector vector multiplication using TBB*/
void
par_matrix_vector_multiply ()
{
	ParVectorMult pmat;
	pmat.vecSize = vecSize;
	pmat.nRows = nRows;
	pmat.nColms = nColms;
	pmat.mA = mA;
	pmat.vB = vB;
	pmat.vR = vR;
	parallel_for (tbb::blocked_range<size_t> (0, nRows, 1000), pmat);
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
	nColms = nRows = vecSize;
 
	tbb::task_scheduler_init init; 

	/*mappping vectors*/
	mA = map_matrix (mA, nRows, nColms);
	vB = map_vector (vB, vecSize);
	vR = map_vector (vR, vecSize);
  
	/* filling vectors*/
	fill_matrix (mA);
	fill_vector (vB);

//	print_matrix(mA);
//	print_vector(vB);

	/*Vector vector multiplication */
	tick_count t0 = tick_count::now();
	par_matrix_vector_multiply ();
	tick_count t1 = tick_count::now();
	
	/*Results*/
	printf("\n\t\t Matrix-Vector Multiplication Done ..!!");
	printf("\n\t\t Matrix Size: %ld",vecSize);
	printf("\n\t\t Time taken to perform: %lf Sec \n",(t1-t0).seconds() );
	for (int i = 0; i < vecSize; ++i){
	printf("\t %lf", vR[i]);
	}
	
	/*Freeing memory*/
	free_memM (mA);
	free_memV (vB);
	free_memV (vR);
	printf("\n\t\t Memory freed up successfully ..!!\n\n");
}
