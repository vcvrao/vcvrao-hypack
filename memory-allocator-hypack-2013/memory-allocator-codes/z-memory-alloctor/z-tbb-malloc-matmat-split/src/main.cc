********************************************************************

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

size_t nrows, ncols;
float **ma, **mb, **mc,**ms;

/* main function start here */
using namespace tbb;

int main (int argc, char **argv)
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


 // printf (" Matrix Filled \n");

   /* Allocate the memory for matrics */

  ma = map_matrix (ma,nrows,ncols);
  mb = map_matrix (mb,nrows,ncols);
  mc = map_matrix (mc, nrows, ncols);

  /* preparing the input */

  fill_matrix (ma);
  fill_matrix (mb);
  fill_matrix (mc);
//  print_matrix (ma);
//  print_matrix (mb);
  tick_count t0 = tick_count::now();

 par_matrix_matrix_multiply ();
  tick_count t1 = tick_count::now();
  double tm=(t1-t0).seconds();

//  print_matrix (mc);



  printf (" THREADS\t  TIME(PARLLEL EXE IN SECOND)\n");
  printf ("-------------------------------------------- \n");
  printf (" %d\t\t  %g \n",numThreads,tm);
  printf("\n");

 /* Free the memory */

  free_mem(ma,nrows,ncols);
  free_mem(mb,nrows,ncols);
  free_mem(mc,nrows,ncols);

}

