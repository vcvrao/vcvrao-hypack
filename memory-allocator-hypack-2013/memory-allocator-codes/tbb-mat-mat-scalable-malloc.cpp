
/**************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

	Example    : TbbMatMatScalableMalloc.cpp
		     using TBB LIBRARY API and scalable_malloc memory.

       Demonstrate : parrel_for().
		     blocked_range2d().   
		     scalable_malloc().
		     scalable_free()	

	Input      : Matrix size.

	Output     : Product of Matrix-Matrix, Time in seconds.

      Created      : August-2013

      E-mail       : hpcfte@cdac.in     

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
#include <tbb/blocked_range2d.h>
#include <tbb/scalable_allocator.h>

using namespace tbb;
using namespace std;

/* Global variable declaration*/
size_t nRows, nColms; 
float **mA, **mB, **mR;
size_t mapsize;
size_t mapCsize;

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
void free_mem(float **matrix)
{
	scalable_free (matrix);
}

/* Structure for Matrix Matrix multiplication using TBB*/
struct ParMatrixMult
{
	size_t nRows, nColms;
	float **mR,**mA, **mB;
	void operator () (const blocked_range2d < size_t, size_t> &r) const{
	int i, j, k ;
	float sum ;
    	for (i = r.rows().begin (); i < r.rows().end (); ++i){
    	for (j = r.cols().begin (); j < r.cols().end (); ++j){
		sum = 0.0;
	for (k = 0; k < nColms; ++k){
		sum += mA[i][k] * mB[k][j];
	}
		mR[i][j] = sum;
        }	
	}
	}	
};


/* Matrix Matrix multiplication using TBB*/
void
par_matrix_matrix_multiply ()
{
	ParMatrixMult pmat;
	pmat.nRows = nRows;
	pmat.nColms = nColms;
	pmat.mA = mA;
	pmat.mB = mB;
	pmat.mR = mR;
	parallel_for (tbb::blocked_range2d<size_t, size_t > (0, nRows, 0, nColms), pmat);
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
		printf ("\n Usage: executable # Matrix size \n");
		exit (-1);
	}

	/* User input taking*/	
	nRows = atoi (argv[1]);
	nColms = nRows;
 
	tbb::task_scheduler_init init; 

	/*mappping Matrixs*/
	mA = map_matrix (mA, nRows, nColms);
	mB = map_matrix (mB, nRows, nColms);
	mR = map_matrix (mR, nRows, nColms);
  
	/* filling Matrixs*/
	fill_matrix (mA);
	fill_matrix (mB);

//	print_matrix(mA);
//	print_matrix(mB);

	/*Matrix Matrix multiplication */
	tick_count t0 = tick_count::now();
	par_matrix_matrix_multiply ();
	tick_count t1 = tick_count::now();
	
	/*Results*/
	printf("\n\t\t Matrix-Matrix Multiplication Done ..!!");
	printf("\n\t\t Matrix Size: %ld",nRows);
	printf("\n\t\t Time taken to perform: %lf Sec \n",(t1-t0).seconds() );
	for (int i = 0; i < nRows; ++i){
	for (int j = 0; j < nColms; j++){
	printf("\t %lf", mR[i][j]);
	}
	}	
	/*Freeing memory*/
	free_mem (mA);
	free_mem (mB);
	free_mem (mR);
	printf("\n\t\t Memory freed up successfully ..!!\n\n");
}
