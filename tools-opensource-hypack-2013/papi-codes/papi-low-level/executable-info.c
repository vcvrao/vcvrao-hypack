/**************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example          : executable-info.c

 Objective        : Program to get the info of Executable file. 
	       	    const PAPI_exe_info_t *PAPI_get_executable_info(void)
	    
	  typedef struct _papi_program_info {
      	char fullname[PAPI_HUGE_STR_LEN];  // path+name 
      	PAPI_address_map_t address_info;
     	 } PAPI_exe_info_t;

	 typedef struct _papi_address_map {
      	char name[PAPI_HUGE_STR_LEN];
      	caddr_t text_start;       // Start address of program text segment 
      	caddr_t text_end;         // End address of program text segment   
      	caddr_t data_start;       // Start address of program data segment   
      	caddr_t data_end;         // End address of program data segment   
      	caddr_t bss_start;        // Start address of program bss segment   
      	caddr_t bss_end;          // End address of program bss segment   
  		 } PAPI_address_map_t;

 Input                : None

 Output               : Displays the information of the executable.


  Created             : August-2013

  E-mail              : hpcfte@cdac.in     


*****************************************************************************/
/*
 Program Name : executable-info.c 
 
 Description : Program to get the info of Executable file. 
	       const PAPI_exe_info_t *PAPI_get_executable_info(void)
    
  typedef struct _papi_program_info {
      char fullname[PAPI_HUGE_STR_LEN];  // path+name 
      PAPI_address_map_t address_info;
      } PAPI_exe_info_t;

 typedef struct _papi_address_map {
      char name[PAPI_HUGE_STR_LEN];
      caddr_t text_start;       // Start address of program text segment 
      caddr_t text_end;         // End address of program text segment   
      caddr_t data_start;       // Start address of program data segment   
      caddr_t data_end;         // End address of program data segment   
      caddr_t bss_start;        // Start address of program bss segment   
      caddr_t bss_end;          // End address of program bss segment   
   } PAPI_address_map_t;


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
	  const PAPI_exe_info_t *exe_file_info = NULL;
  	  static int bss_var=0;	

	  
	   printf("\n  **********************************************************************************\n");	
           printf("\n\t Description : Program to get the info of Executable file  \n ");
           printf("\n");
	   printf("\n  **********************************************************************************\n");	

 	  printf("\n\t=> PAPI Library initialization : ");
	  if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT )
   	   {
      	    printf("\n\t Error : PAPI Library initialization error! \n");
     	    return(-1);
	   }
	  printf("Done");          

            // Doing some computation here (square matrix-matrix multiplication )... //
	    matrix_matrix_multiply();
	   
 	  printf("\n\t=> Getting the executable file info : ");
	   if((exe_file_info = PAPI_get_executable_info()) == NULL)
	 	 {   printf("\n\t   Error : PAPI failed to get executable file info. \n");
                     printf("\n\t   Error string : %s  :: Error code : %d \n",PAPI_strerror(retval),retval);
                  return(-1);   }
	  printf("Done");

	   printf("\n\t Executable info ");
	   printf("\n\t\t Executable file name : %s", exe_file_info->fullname);
	   printf("\n\t\t Start address of program text segment : %p", exe_file_info->address_info.text_start);
	   printf("\n\t\t End address of program text segment : %p", exe_file_info->address_info.text_end);
	   printf("\n\t\t Start address of program data segment : %p", exe_file_info->address_info.data_start);
	   printf("\n\t\t End address of program data segment : %p", exe_file_info->address_info.data_end);
	   /*printf("\n\t\t Start address of program bss segment : %p", exe_file_info->address_info.bss_start);
	   printf("\n\t\t End address of program bss segment : %p", exe_file_info->address_info.bss_end);*/

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
                	{ for(i=0;i<size;i++)
        	             {  mat_c[irow][icol] = mat_c[irow][icol] + (mat_a[irow][i]*mat_b[i][icol]);  } } } 
	return mat_c; }



