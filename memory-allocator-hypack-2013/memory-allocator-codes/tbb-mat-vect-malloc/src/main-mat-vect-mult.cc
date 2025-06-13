
/********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example                : main-mat-vect-mult.cc


 Objective  		: To perform matrix-vector multiplication
                          using TBB.Demonstrates use of tbb
                          parallel_for().
 
 Input                  : executable #  size # Number Of Threads 

 Output                 : execution time in seconds	                                            
                                                                        
 Created                : August-2013

 E-mail                 : hpcfte@cdac.in     

*************************************************************************/


#include"../include/headerfiles.h"
#include"../include/ParMatVectMult.h"
#include"../include/proto.h"


int nrows,ncols,vsize;

float *matrixA,*vectorA,*result_vector;

int main (int argc, char **argv)
{
 
  if (argc != 3)                                                         // check command line input        
    {
      printf ("Usage: executable #  size # Number Of Threads \n");       
      return 1;
    }

  int numThreads = atoi(argv[2]);
  tbb::task_scheduler_init init (numThreads);

  nrows = atoi (argv[1]);
  ncols = nrows;
  vsize = ncols;


  mat_vec_memory_allocation(&matrixA,&vectorA,&result_vector); // allocate memory

  printf("\n Memory Allocated successfully..........");

  matrix_vector_input (matrixA,vectorA);                        // give input to matrix and vector

   tick_count t0 = tick_count::now();

  par_matrix_vector_multiply ();                                // call function to perform multiplication

  tick_count t1 = tick_count::now();

  double t_parallel = (t1-t0).seconds();                       // calculate time for execution


  print_output(nrows,numThreads,t_parallel);                  // call function to print output

  
  memoryfree(matrixA,vectorA,result_vector);                 // call function to free memory

  return 0;

}
