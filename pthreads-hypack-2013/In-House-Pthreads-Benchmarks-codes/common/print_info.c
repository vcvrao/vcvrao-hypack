/*
  print_info.c
  
  Utility to print the compiler, libraries other information used in the compilation
*/  


#include <stdlib.h>
#include <stdio.h>

void print_info( char   *name,
                      char   class,
                      int    threads,
                      double t,
                      char   *compile_time,
                      char   *cc,
                      char   *clink,
                      char   *c_lib,
                      char   *c_inc,
                      char   *cflags,
                      char   *clinkflags )
{
   /* printf( "\n\n\t\t :: SUMMARY :: " ); 
    printf( "\n\t\t %s Benchmark Completed", name ); 
    printf( " \n\t\t Class             = %c", class );
    printf( " \n\t\t Time in seconds   = %.2f", t );
    printf( " \n\t\t Number of Threads = %d", threads); */

    printf( "\n\t\t Summary of Compiler and Libraries used:" );
    printf( "\n\t\t Compiled on  = %s", compile_time );
    printf( "\n\t\t CC           = %s", cc );
    printf( "\n\t\t CLINK        = %s", clink );
    printf( "\n\t\t C_LIB        = %s", c_lib );
    printf( "\n\t\t C_INC        = %s", c_inc );
    printf( "\n\t\t CFLAGS       = %s", cflags );
    printf( "\n\t\t CLINKFLAGS   = %s", clinkflags );
    printf("\n\t\t..........................................................................\n\n");
}
 
