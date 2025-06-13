#include"../include/headerfiles.h"
#include"../include/proto.h"

void print_output(int nrows,int numThreads,double t_parallel)   // print result 
{
  printf("\n");
  printf("\n Size = %d\n",nrows);
  printf("\n NumThreads = %d \n",numThreads);
  printf("\n Execution Time = %g\n ",t_parallel);
}
