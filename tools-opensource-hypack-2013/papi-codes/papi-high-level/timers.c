/*******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example              : timers.c


 Objective            : UProgram to use the PAPI real and Virtual Timers. 
		Real Timers :
			long_long PAPI_get_real_cyc(void)
			long_long PAPI_get_real_usec(void)
		Virtual Timers :
			long_long PAPI_get_virt_cyc(void)
			long_long PAPI_get_virt_usec(void)

 Input                : None

 Output               : Displays the real and virtual timings of 
			matrix-matrix multiplication.

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

*****************************************************************************/
/*
 Program Name : timers.c
 
 Description : Program to use the PAPI real and Virtual Timers. 
	Real Timers :
		long_long PAPI_get_real_cyc(void)
		long_long PAPI_get_real_usec(void)
	Virtual Timers :
		long_long PAPI_get_virt_cyc(void)
		long_long PAPI_get_virt_usec(void)
 
*/

#include<stdio.h>
#include<stdlib.h>
#include "papi.h"

int** my_generate_matrix(int, int**);
int** my_matrix_multiply(int, int **, int **, int **);
void matrix_matrix_multiply(void);

int main()
	{
	  int retval;
	  long_long start_real_usec, start_real_cyc, start_virt_usec, start_virt_cyc;
	  long_long end_real_usec, end_real_cyc, end_virt_usec, end_virt_cyc;

	   printf("\n  **********************************************************************************\n");	
           printf("\n\t Description : Program using the PAPI Real and Virtual timers  \n ");
	   printf("\n\t\t  PAPI_get_real_cyc(), PAPI_get_real_usec() \n\t\t PAPI_get_virt_cyc(), PAPI_get_virt_usec()");
           printf("\n");
	   printf("\n  **********************************************************************************\n");	
 	
	  if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
   	   {
      	    printf("\n\t Error : PAPI Library initialization error! \n");
     	    return(-1);
	   }
          
	   start_real_usec = PAPI_get_real_usec();	 
	   start_real_cyc = PAPI_get_real_cyc();	 
	   start_virt_usec = PAPI_get_virt_usec();	 
	   start_virt_cyc = PAPI_get_virt_cyc();	 

      	    printf("\n\t Matrix-Matrix Multiplication ");
            // Doing some computation here (square matrix-matrix multiplication )... //
	    matrix_matrix_multiply();

	   end_real_usec = PAPI_get_real_usec();	 
	   end_real_cyc = PAPI_get_real_cyc();	 
	   end_virt_usec = PAPI_get_virt_usec();	 
	   end_virt_cyc = PAPI_get_virt_cyc();	 

	   printf("\n\t Real (wall) Time taken (in micro seconds) :%ld ",(end_real_usec - start_real_usec));
	   printf("\n\t Real (wall) Time taken (in cycles) :%ld ",(end_real_cyc - start_real_cyc));
	   printf("\n\t Virtual Time taken (in micro seconds) :%ld ",(end_virt_usec - start_virt_usec));
	   printf("\n\t Virtual Time taken (in cycles) :%ld ",(end_virt_cyc - start_virt_cyc));
	   PAPI_shutdown();

   	printf("\n-----------------------------------------------------------------------------\n");
	return 0;
	}

/*
 Function : matrix_matrix_multiply()
 Author: Shiva  Date : Dec 13 2008
*/


void matrix_matrix_multiply(void)
 {
  int irow,icol,i;
  int mat_size;	
  int **mat_a,**mat_b,**mat_c;
  srand((unsigned)time(NULL)); 
  mat_size = 100;
  printf("\n\t Size of matrix taken as %d",mat_size);

    /* Memory Allocation for Matrix A */
    printf("\n\t Generating the Matrix A and Matrix B with size %d.",mat_size);	
    mat_a = my_generate_matrix(mat_size, mat_a);
    mat_b = my_generate_matrix(mat_size, mat_b);
    
    /* Matrix Multiplication  Matrix C = Matrix A * Matrix B*/
    printf("\n\t Matrix multiplication :");	
    mat_c = my_matrix_multiply(mat_size, mat_a, mat_b, mat_c);
    printf(" Done ");	

  /* Freeing the Memory ... */
  free(mat_a);  free(mat_b);  free(mat_c);  } 


int** my_generate_matrix(int size, int **mat)
    {     int irow,icol,i;
    /* Memory Allocation for Matrix and populating matrix with randomly generated number */
    mat = (int **)malloc(size * sizeof(int));
    for(irow = 0; irow < size; irow++)
    	{  	mat[irow] = (int *)malloc(size * sizeof(int));	
     	  for(icol = 0; icol < size; icol++)
      		{      mat[irow][icol] = rand()%10;		}    	}
	return mat;    }

int** my_matrix_multiply(int size, int **mat_a, int **mat_b, int **mat_c)
	{     	int irow,icol,i;
        /* Allocating the memory for the resultant */ 
    	mat_c = (int **)malloc(size * sizeof(int));
	    for(irow = 0; irow < size; irow++)
    		{ mat_c[irow] = (int *)malloc(size * sizeof(int));	
     		for(icol = 0; icol < size; icol++)
      			{      mat_c[irow][icol] = 0;		}  	}
	 /* Matrix Multiplication  Matrix C = Matrix A * Matrix B*/
	for(irow = 0; irow < size; irow++)
	        {     	for(icol = 0; icol < size; icol++)
                	{ for(i=0;i<s
