#include"../include/headerfiles.h"

void print_output(int vsize,int numThreads,double t_parallel)   // print result 
{
  printf("\n");
  printf("\n Size = %d\n",vsize);
  printf("\n NumThreads = %d \n",numThreads);
  printf("\n Execution Time = %g\n ",t_parallel);
}
