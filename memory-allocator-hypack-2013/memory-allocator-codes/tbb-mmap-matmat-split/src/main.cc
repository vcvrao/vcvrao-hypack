/********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example                : main.cc(for matrix multiplication)


 Objective              : To perform matrix-matrix multiplication
                          using TBB and mmap memory allocator.

Demonstrates use        :  parallel_for().

 Input                  : executable #  size # Number Of Threads

 Output                 : execution time in seconds

 Created                : August-2013

 E-mail                 : hpcfte@cdac.in     

***********************************************************************/




#include "../include/define.h"
#include "../include/sysheader.h"
#include "../include/proto.h"

 int fda, fdb, fdc,fds;
 size_t nrows, ncols;
 float *ma, *mb, *mc,*ms;
 size_t mapsize;

/* main function start here */
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
/* prepare the input */

  fill_matrix ("./tbb-input/matrixA");
  fill_matrix ("./tbb-input/matrixB");
  fill_matrix ("./tbb-input/matrixC");

 // printf (" Matrix Filled \n");

 /*maping the input file to process address space 
  using mmap memory allocator */

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

/* cloesing the file descriptor */ 

  close (fda);
  close (fdb);
  close (fdc);
// close (fds);
}

