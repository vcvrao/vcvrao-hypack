/**************************************************************************

     C-DAC Tech Workshop : hyPACK-2013
              October 15-18, 2013


    Example       : main-vect-vect-mult.cc

    Objective     : To get the Performance using Intel TBB for the 
                    vector vector multiplication using mmap
                    Demonstrates usse of: parallel_for() 
     
    Input         : Vector size and Number of threads

    Output        : Time taken to compute the vector vector multiplication

    Created       : August-2013

    E-mail        : hpcfte@cdac.in     

*************************************************************************/

#include"../include/headerfiles.h"
#include"../include/ParVectVectMult.h"
#include"../include/proto.h"               // header file inclusion
#include"../include/define.h"

int vsize;                                 // variable declaration


float *vectorB;
float *vectorA;
float *result_vector;

int fda,fdb,fdc;

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

  vsize = atoi (argv[1]);


  vectorB = mmap_vec_mem_allocation("./tbb-input/vectorB", MAP_RDONLY, vsize, &fda);      // call mem allocation
  vectorA = mmap_vec_mem_allocation("./tbb-input/vectorA", MAP_RDONLY, vsize, &fdb);      //function
  result_vector = mmap_vec_mem_allocation("./tbb-input/result_vector", MAP_RDWR, vsize, &fdc);

  printf("\n Memory Allocated successfully........");  

  mmap_vector_input ("./tbb-input/vectorB");
  mmap_vector_input ("./tbb-input/vectorA");                                // give input to vector and vector
  


  tick_count t0 = tick_count::now();                                         // time calculation

  par_vector_vector_multiply ();                               // call vec-vec multiplication function

  tick_count t1 = tick_count::now();

  double t_parallel = (t1-t0).seconds();

  print_output(vsize,numThreads,t_parallel);                 // call a function to print result

  size_t mapvsize = vsize * sizeof(float);
  memoryfree(vectorB,vectorA,result_vector,mapvsize);        // call a function to free memory

  close (fda);
  close (fdb);
  close (fdc);

  return 0;
}
