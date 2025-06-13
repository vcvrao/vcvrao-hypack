/*********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

    Example       : tbb-vectorvector.cpp

    Objective     : To get the Performance using Intel TBB 
			Demonstrates usse of:
				parallel_for() 
     
    Input 	  : Vector size and Number of threads

    Output        : Time taken to compute the vector vector multiplication

   Created        : August-2013

   E-mail         : hpcfte@cdac.in     

*******************************************************************/

/*Header files of cpp */
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

/*Header files of TBB*/
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

/* Macros */
#define MAP_RDONLY   1
#define MAP_RDWR     2

using namespace tbb;
using namespace std;


/* Global declearation*/
int fda, fdb, fdc;
size_t vecSize; 
float *va, *vb, *vc;
size_t mapsize;


inline size_t
getID (int i)
{
	assert (i >= 0 && i < vecSize);
	return  i;
}

inline float
getVal (const float *buf, int i)
{
	size_t id = getID (i);
	return buf[id];
}

inline void
setVal (float *buf, int i, float val)
{
	size_t id = getID (i);
	buf[id] = val;
}


/*Input Vector filling*/
void
fill_vector (char *fname)
{	/*opening a input file*/
	FILE *fp = fopen (fname, "w+");
	if (fp == NULL)
	{
	printf (" Cann't open the file: %s \n", fname);
	return;
	}

	int counter = 1;
	int nwrite;
	
	for (size_t i = 0; i < vecSize; i++)
	{
	size_t id = getID (i);
	float val = id;
	nwrite = fwrite (&val, sizeof (float), 1, fp);
	assert (nwrite == 1);
	}
 
	fclose (fp);
}


/*Printing the Input vector file*/
void
print_file (const char *fname)
{
	FILE *fp = fopen (fname, "r");
	if (fp == NULL)
	{
	printf (" Cann't open the file: %s \n", fname);
	return;
	}

	float val;
	for (size_t i = 0; i < vecSize; i++)
	{
	fread (&val, sizeof (float), 1, fp);
	printf ("%lf ", val);
	}
	printf ("\n");
    
}


/*Printing the vector*/
void
print_vector(const float *buf)
{
	float val;
	for (size_t i = 0; i < vecSize; i++)
	{
	val = getVal (buf, i);
	printf ("%lf ", val);
	}
	printf ("\n");
}

/* Memory Mapping of a vector */
float *
map_vector (char *fname, int mode, int vecSize, int *fileid)
{
	int fd, stat;
	size_t mapsize;
	float *map_addr = NULL;

	mapsize = vecSize * sizeof (float);
	if (mode == MAP_RDONLY)
	fd = open (fname, O_RDONLY);

	if (mode == MAP_RDWR)
	fd = open (fname, O_RDWR);

	if (fd <= 0)
	{
	printf ("Error: Cann't open file vectorA\n");
	return NULL;
	}	

	if (mode == MAP_RDONLY)
	map_addr = (float *) mmap (0, mapsize, PROT_READ, MAP_SHARED, fd, 0);

	if (mode == MAP_RDWR)
	map_addr =
	 (float *) mmap (0, mapsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

	if (map_addr == MAP_FAILED)
	{
	printf ("Error: mmap failed \n");
	exit (0);
	return NULL;
	}
	*fileid = fd;

	return map_addr;
}

/*sequetial vector vector multiply*/
void
seq_vector_vector_multiply ()
{
	int i;
	float sum, ai, bi;
	printf (" Multiplication Starts ... \n");
	for (i = 0; i < vecSize; i++)
	{
	sum = 0.0;
	ai = getVal (va, i);
	bi = getVal (vb, i);
	sum += ai * bi;
   
	setVal (vc, i, sum);
	}
    
	printf (" Multiplication Ends ... \n");
}

/* Class */
struct ParVectorMult
{
	size_t vecSize;
	float *vc, *va, *vb;

void operator () (const blocked_range < size_t > &r) const
{
	int i,j;
	float ai, bi, sum;
	for (j = r.begin (); j != r.end (); ++j)
	{
	for (i = 0; i < vecSize; i++)
	{
	sum = 0.0;
	
	ai = getVal (va, i);
	bi = getVal (vb, i);
	sum += ai * bi;
			              
	setVal (vc, i, sum);
	}
	}
//	printf("%lf\t",sum);  
}
};


/*Parallel vector multiplication*/
void
par_vector_vector_multiply ()
{
	ParVectorMult pmat;
	pmat.vecSize = vecSize;
	pmat.va = va;
	pmat.vb = vb;
	pmat.vc = vc;
	/*Template function*/
	parallel_for (tbb::blocked_range<size_t> (0, vecSize), pmat);
}


/*Main function*/
int
main (int argc, char **argv)
{

	/*Time Template */
	tick_count t0 = tick_count::now();
	if (argc != 3)
	{
	printf ("Usage: executable # vector size # No of Threads\n");
	return 1;
	}

	int numThreads;
	
	numThreads = atoi (argv[2]);
	
	/*TBB Template Initialization*/
	tbb::task_scheduler_init init (numThreads);
  

	vecSize = atoi (argv[1]);
	mapsize = vecSize * sizeof (float);

	/*Input vector filling*/
	fill_vector ("./tbb-input/vectorA");
	fill_vector ("./tbb-input/vectorB");
	fill_vector ("./tbb-input/vectorC");

	printf (" Vector Filled \n");
	
	/*Mapping memory */
	va = map_vector ("./tbb-input/vectorA", MAP_RDONLY, vecSize, &fda);
	vb = map_vector ("./tbb-input/vectorB", MAP_RDONLY, vecSize, &fdb);
	vc = map_vector ("./tbb-input/vectorC", MAP_RDWR, vecSize, &fdc);

	/*print_vector(va);
     	print_vector(vb);
     	print_vector(vc);*/

	/*Calling the parllel multiplication function*/
	par_vector_vector_multiply ();

	/*Unmapping memory */
	munmap (va, mapsize);
	munmap (vb, mapsize);
	munmap (vc, mapsize);

	printf (" Vector Multiplication Done  \n");
	
	/*Closing the files*/
	close (fda);
	close (fdb);
	close (fdc);

	/*Time Template*/
	tick_count t1 = tick_count::now();


	printf("\n Number Threads\t time(Sec)");
	printf("\n %d\t %lf \n",numThreads,(t1-t0).seconds() );
}
