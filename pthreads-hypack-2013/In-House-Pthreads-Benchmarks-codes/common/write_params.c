/* 
 * The utility takes 3 arguments: 
 *       setparams benchmark-name nthreads class
 *    benchmark-name is either "pi, mat-mat-db, mat_mat_in, jacobi"   
 *    nprocs is the number of processors to run on
 *    class is the size of the benchmark
 * These parameters are checked for the current benchmark. If they
 * are invalid, this program prints a message and aborts. 
 * If the parameters are valid, the file params.h is generated with resopective 
 * benchmark.
 * Everything has to be case sensitive. So typing make CLASS=a and make CLASS=A 
 * will produce different binaries.
 */

#include <stdio.h>
#include "params_lib.h"

 int main(int argc, char *argv[])
     {
      int nthreads;
      char class,benchmark[12];
      FILE *fp; 
      /* verify the command line arguments.*/
      get_args(argc,argv,&nthreads ,&class, benchmark);
      check_class(class);

      fp = fopen(FILENAME, "w");
      if(fp == NULL)
       {
        printf("Error : Failed to generate header file for %s \n", benchmark);
        exit(-1);
       } 
      write_header(fp,benchmark,class,nthreads);
      /*printf("\n Benchmark: %s, Class : %c , # Threads : %d\n",benchmark,class,nthreads);*/
      return 0;
     }




