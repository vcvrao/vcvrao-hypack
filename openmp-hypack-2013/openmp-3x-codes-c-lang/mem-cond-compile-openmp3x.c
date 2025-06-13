/***************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

           OpenMP-3.0 Example Codes Beta-v1.0      
         
File          :  mem-cond-compile-openmp3x.c 

Description   : Simple example program to illustrate the use of 
                openmp conditional compilation _OPENMP macro. 
		
		In the example program print statement will be
                displayed only if the _OPENMP macro is define.
                With OpenMP compilation the _OPENMP macro gets 
                defined. Which can be used to ignored the openMP
                Library calls in the program if the compilation
                is not done with the openMP support.

Input         : None 

Output        : Print statement will be printed if the macro				
		_OPENMP is defined. 

Created       : August-2013

E-mail        : hpcfte@cdac.in     

***********************************************************/

/* Header file inclusion */
#include <stdio.h>
#include<omp.h>
/* Main function */
int main()
 {
 
 /*  Conditional compilation */
 # ifdef _OPENMP
     printf("Compiled by an OpenMP-compliant implementation.\n");
 # endif
     return 0;


 }

