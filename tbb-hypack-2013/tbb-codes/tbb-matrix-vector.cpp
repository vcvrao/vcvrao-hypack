/********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example                : tbb-matrix-vector.cc


 Objective              : To perform matrix-vector multiplication
                          using TBB.Demonstrates use of tbb
                          parallel_for().

 Input                  : executable #  size # Number Of Threads

 Output                 : execution time in seconds

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
#include<tbb/tick_count.h>

#define MAP_RDONLY   1
#define MAP_RDWR     2

using namespace tbb;
using namespace std;

int fda, fdb, fdc;
size_t nrows, ncols,vsize;
float *ma, *vb, *vc;
size_t mapsize,mapsize1;

inline size_t
getMID (int i, int j)
{
  assert (i >= 0 && i < nrows);
  assert (j >= 0 && j < ncols);

  return j * nrows + i;
}

inline size_t
getVID (int i)
{
  assert (i >= 0 && i < nrows);
 
  return i;
}


inline float
getMVal (const float *buf, int i, int j)
{
  size_t id = getMID (i, j);
  return buf[id];
}

inline float
getVVal (const float *buf, int i)
{
  size_t id = getVID (i);
  return buf[id];
}


inline void
setMVal (float *buf, int i, int j, float val)
{
  size_t id = getMID (i, j);
  buf[id] = val;
}
inline void
setVVal (float *buf, int i, float val)
{
  size_t id = getVID (i);
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
	  size_t id = getMID (i, j);
	  float val = id;
	  nwrite = fwrite (&val, sizeof (float), 1, fp);
	  assert (nwrite == 1);
	}
    }
  fclose (fp);
}

void
fill_vector(char *fname)
{
  FILE *fp = fopen (fname, "w+");
  if (fp == NULL)
    {
      printf (" Cann't open the file: %s \n", fname);
      return;
    }

  int counter = 1;
  int nwrite;
      for (size_t i = 0; i <vsize; i++)
        {
          size_t id = getVID (i);
          float val = id;
          nwrite = fwrite (&val, sizeof (float), 1, fp);
          assert (nwrite == 1);
        }
    
  fclose (fp);
}

float *
map_matrix (char *fname, int mode, int nrows, int ncols, int *fileid,int flag)
{
  int fd, stat;
  size_t mapsize;
  float *map_addr = NULL;

  ncols = nrows;
  vsize = ncols;
  if(flag == 1)
    mapsize = nrows * ncols * sizeof (float);
  else 
    mapsize = vsize * sizeof(float);

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


struct ParMatVectMult
{
  size_t nrows;
  size_t ncols;
  size_t vsize;
  float *vc, *ma, *vb;

  void operator () (const blocked_range < size_t > &r) const
  {
    int i, j, k;
    float aij, bj, sum;
    for (i = r.begin (); i != r.end (); ++i)
      {
	    sum = 0.0;
	  for (j = 0; j < ncols; ++j)
	  {
		aij = getMVal (ma, i, j);
		bj = getVVal (vb,j);
		sum += aij * bj;
	  }
	    setVVal (vc,i, sum);
	  
      }
  }
};

void
par_matrix_vector_multiply ()
{
  ParMatVectMult pmat;
  pmat.nrows = nrows;
  pmat.ncols = ncols;
  pmat.vsize = vsize;
  pmat.ma = ma;
  pmat.vb = vb;
  pmat.vc = vc;
  parallel_for (tbb::blocked_range<size_t> (0, nrows), pmat);
}

int
main (int argc, char **argv)
{
  if (argc != 3)
    {
      printf ("Usage: executable #  size # Number Of Threads \n");
      return 1;
    }

  int numThreads = atoi(argv[2]);
  tbb::task_scheduler_init init (numThreads);

  nrows = atoi (argv[1]);
  ncols = nrows;
  vsize = ncols;
  mapsize = nrows * ncols * sizeof (float);

   mapsize1 = vsize * sizeof (float);

  fill_matrix ("./tbb-input/matrixA");
  fill_vector ("./tbb-input/vectorB");
  fill_vector ("./tbb-input/vectorC");
  

  ma = map_matrix ("./tbb-input/matrixA", MAP_RDONLY, nrows, ncols, &fda,1);
  vb = map_matrix ("./tbb-input/vectorB", MAP_RDONLY, nrows, ncols, &fdb,0);
  vc = map_matrix ("./tbb-input/vectorC", MAP_RDWR, nrows, ncols, &fdc,0);
  
  tick_count t0 = tick_count::now();

  par_matrix_vector_multiply ();

  tick_count t1 = tick_count::now();

  double t_parallel = (t1-t0).seconds();

  printf("\n\n");

  printf(" numThreads         Execution Time\n");
  printf("%d     \t  %g",numThreads,t_parallel);

  munmap (ma, mapsize);
  munmap (vb, mapsize1);
  munmap (vc, mapsize1);


  close (fda);
  close (fdb);
  close (fdc);
}
