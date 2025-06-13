/********************************************************************
		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example             : tbb-matmat-multiply.cc

 Objective           : To perform matrix-matrix multiplication
                          using TBB.
 Demonstrates        : parallel_for().

 Input               : executable #  size # Number Of Threads

 Output              : execution time in seconds

 Created             : August-2013

 E-mail              : hpcfte@cdac.in     

*************************************************************************/

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

#include "tbb/task_scheduler_init.h"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include<tbb/tick_count.h>

#define MAP_RDONLY   1
#define MAP_RDWR     2

using namespace tbb;
using namespace std;

int fda, fdb, fdc,fds;
size_t nrows, ncols;
float *ma, *mb, *mc,*ms;
size_t mapsize;

inline size_t
getID (int i, int j)
{
  assert (i >= 0 && i < nrows);
  assert (j >= 0 && j < ncols);

  return j * nrows + i;
}

inline float
getVal (const float *buf, int i, int j)
{
  size_t id = getID (i, j);
  return buf[id];
}

inline void
setVal (float *buf, int i, int j, float val)
{
  size_t id = getID (i, j);
  buf[id] = val;
}

void
fill_matrix (char *fname)
{
  FILE *fp = fopen (fname, "w+");
  if (fp == NULL)
    {
      printf (" Cann't open the file: %s \n", fname);
      return;
    }

  int counter = 1;
  int nwrite;
  for (size_t j = 0; j < ncols; j++)
    {
      for (size_t i = 0; i < nrows; i++)
	{
	  size_t id = getID (i, j);
	  float val = id;
	  nwrite = fwrite (&val, sizeof (float), 1, fp);
	  assert (nwrite == 1);
	}
    }
  fclose (fp);
}

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
  for (size_t i = 0; i < nrows; i++)
    {
      for (size_t j = 0; j < ncols; j++)
	{
	  fread (&val, sizeof (float), 1, fp);
	  printf ("%lf ", val);
	}
      printf ("\n \n");
    }
}

void
print_matrix (const float *buf)
{
  float val;
  for (size_t i = 0; i < nrows; i++)
    {
      for (size_t j = 0; j < ncols; j++)
	{
	  val = getVal (buf, i, j);
	  printf ("%lf ", val);
	}
      printf ("\n \n");
    }
}


float *
map_matrix (char *fname, int mode, int nrows, int ncols, int *fileid)
{
  int fd, stat;
  size_t mapsize;
  float *map_addr = NULL;

  ncols = nrows;
  mapsize = nrows * ncols * sizeof (float);

  if (mode == MAP_RDONLY)
    fd = open (fname, O_RDONLY);

  if (mode == MAP_RDWR)
    fd = open (fname, O_RDWR);

  if (fd <= 0)
    {
      printf ("Error: Cann't open file matrixA \n");
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

struct ParMatrixMult
{
  size_t nrows;
  size_t ncols;
  float *mc, *ma, *mb;

  void operator () (const blocked_range2d < size_t > &r) const
  {
    int i, j, k;
    float aik, bkj, sum;
    for (j = r.cols().begin (); j != r.cols().end (); ++j)
      {
	for (i = r.rows().begin(); i!=r.rows().end(); i++)
	  {
	    sum = 0.0;
	    for (k = 0; k < ncols; k++)
	      {
		aik = getVal (ma, i, k);
		bkj = getVal (mb, k, j);
		sum += aik * bkj;
	      }
	    setVal (mc, i, j, sum);
	  }
      }
  }
};

void
par_matrix_matrix_multiply ()
{
  ParMatrixMult pmat;
  pmat.nrows = nrows;
  pmat.ncols = ncols;
  pmat.ma = ma;
  pmat.mb = mb;
  pmat.mc = mc;
  parallel_for (tbb::blocked_range2d<size_t> (0, nrows,100,0,ncols,100), pmat);
}

int
main (int argc, char **argv)
{
  if (argc != 3)
    {
      printf ("Usage: executable # matrix size #numThread \n");
      return 1;
    }

  int numThreads = 1;
  numThreads = atoi (argv[2]);
  tbb::task_scheduler_init init (numThreads);

  nrows = atoi (argv[1]);
  ncols = nrows;
  mapsize = nrows * ncols * sizeof (float);

  fill_matrix ("./tbb-input/matrixA");
  fill_matrix ("./tbb-input/matrixB");
  fill_matrix ("./tbb-input/matrixC");

 // printf (" Matrix Filled \n");

  ma = map_matrix ("./tbb-input/matrixA", MAP_RDONLY, nrows, ncols, &fda);
  mb = map_matrix ("./tbb-input/matrixB", MAP_RDONLY, nrows, ncols, &fdb);
  mc = map_matrix ("./tbb-input/matrixC", MAP_RDWR, nrows, ncols, &fdc);
 

  tick_count t0 = tick_count::now();
  par_matrix_matrix_multiply ();
  tick_count t1 = tick_count::now();

  double tm=(t1-t0).seconds();

  printf (" THREADS\t  TIME(PARLLEL EXE IN SECOND)\n");
  printf ("-------------------------------------------- \n");
  printf (" %d\t\t  %g \n",numThreads,tm);
  printf("\n");
 
  munmap (ma, mapsize);
  munmap (mb, mapsize);
  munmap (mc, mapsize);



  close (fda);
  close (fdb);
  close (fdc);

}
