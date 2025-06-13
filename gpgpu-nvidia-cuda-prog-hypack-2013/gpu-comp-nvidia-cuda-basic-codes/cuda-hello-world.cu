
/*************************************************************************

             C-DAC Tech Workshop : hyPACK-2013
                   October 15-18, 2013

  Example     :   cuda-hello-world.cu
 
  Objective   : Write a CUDA  program for hello world.                 

  Input       : None 

  Output      : printing the "hello world". 
                                                                                                                            
  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

************************************************************************/

#include<cuda.h>
#include <stdio.h>

__global__ void kernel( void ) {
}

int main( void ) {
    kernel<<<1,1>>>();
    printf( "Hello, World!\n" );
    return 0;
}
