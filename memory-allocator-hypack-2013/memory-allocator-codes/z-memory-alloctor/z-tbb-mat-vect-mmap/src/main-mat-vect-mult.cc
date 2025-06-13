
/********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example                : main-mat-vect-mult.cc


 Objective              : To perform matrix-vector multiplication
                          using TBB.Demonstrates use of tbb
                          parallel_for().

 Input                  : executable #  size # Number Of Threads

 Output                 : execution time in seconds

 Created                : August-2013

 E-mail                 : hpcfte@cdac.in     

*************************************************************************/

#include"../include/headerfiles.h"
#include"../include/ParMatVectMult.h"
#include"../include/proto.h"                      // header file inclusion
#include"../include/define.h"

int nrows;
int ncols;
int vsize;                              // variable declaration

float *matrixA;
float *vectorA;
float *result_vector;

int fda, fdb, fdc;

int main (int argc, char **argv)
{
 
 int flag;

 if (argc != 3)                                                            // check user input
    {
      printf ("Usage: executable #  size # Number Of Threads \n");
      return 1;
    }

  int numThreads = atoi(argv[2]);
  tbb::task_scheduler_init init (numThreads);

  nrows = atoi (argv[1]);
  ncols = nrows;
  vsize = ncols;


  matrixA = mmap_matvec_mem_allocation("./tbb-input/matrixA", MAP_RDONLY, nrows, ncols, &fda,0);      // call mem allocation
  vectorA = mmap_matvec_mem_allocation("./tbb-input/vectorA", MAP_RDONLY, nrows, ncols, &fdb,1);      //function
  result_vector = mmap_matvec_mem_allocation("./tbb-input/result_vector", MAP_RDWR, nrows, ncols, &fdb,1);

  printf("\n Memory Allocated successfully........");  

  mmap_matrix_input ("./tbb-input/matrixA");
  mmap_vector_input ("./tbb-input/vectorA");                                // give input to matrix and vector
  mmap_vector_input ("./tbb-input/result_vector");


  tick_count t0 = tick_count::now();                                         // time calculation

  par_matrix_vector_multiply ();                               // call mat-vec multiplication function

  tick_count t1 = tick_count::now();

  double t_parallel = (t1-t0).seconds();

  print_output(nrows,numThreads,t_parallel);                 // call a function to print result

  size_t mapvsize = vsize * sizeof(float);
  size_t mapmsize = nrows * ncols * sizeof(float);
  memoryfree(matrixA,vectorA,result_vector,mapmsize,mapvsize);        // call a function to free memory

  close (fda);
  close (fdb);
  close (fdc);

  return 0;
}
