/**************************************************************************

    C-DAC Tech Workshop : hyPACK-2013
           October 15-18, 2013

     Example       : main-vect-vect-mult.cc

     Objective     : To get the Performance for the vector vector multiplication 
                        using Intel TBB with malloc()
                        Demonstrates usse of: parallel_for() 
     
     Input         : Vector size and Number of threads

     Output        : Time taken to compute the vector vector multiplication

     Created       : August-2013

     E-mail        : hpcfte@cdac.in     

****************************************************************************/


#include"../include/headerfiles.h"
#include"../include/ParVectVectMult.h"
#include"../include/proto.h"
//#include"../include/functionfiles.h"

int vsize;                              // variable declaration

float *vectorB;
float *vectorA;
float *result_vector;

//using namespace tbb;
int main (int argc, char **argv)
{
 
  if (argc != 3)                     // check command line input        
    {
      printf ("Usage: executable #  size # Number Of Threads \n");       
      return 1;
    }

  int numThreads = atoi(argv[2]);
  tbb::task_scheduler_init init (numThreads);

  vsize = atoi (argv[1]);

  vec_memory_allocation(&vectorB,&vectorA,&result_vector); //memory allocation for vector and vector

  printf("\n Memory Allocated successfully..........");

  vector_input (vectorB,vectorA);                        // give input to vector vector

   tick_count t0 = tick_count::now();
//   tbb::tick_count t0 = tbb::tick_count::now();

  par_vector_vector_multiply ();                     // call function to perform multiplication

  tick_count t1 = tick_count::now();
 // tbb::tick_count t1 = tbb::tick_count::now();

  double t_parallel = (t1-t0).seconds();               // calculate time for execution


  print_output(vsize,numThreads,t_parallel);          // call function to print output

  
  memoryfree(vectorB,vectorA,result_vector);          // call function to free memory

  return 0;

}
