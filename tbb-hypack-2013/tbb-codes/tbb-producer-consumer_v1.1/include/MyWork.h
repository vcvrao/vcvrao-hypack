#ifndef __MYWORK_H__
#define __MYWORK_H__

#include"../include/headerfiles.h"
#include"../include/fun_declaration.h"

using namespace std;
using namespace tbb;

// structure for giving work to each consumer
struct MyWork       
{
   double **mat1;                                       
   double **mat2;
   double **result_mat;
   int size;
   int nrows,ncols;
   int i,j,k;

   MyWork(int nsize)
   {
     nrows = nsize;
     ncols = nsize;
   }

   MyWork()
   {
	nrows = 50;
        ncols = nrows;
   }
                                                         
  // each producer generates data for mat-mat multiplication
  void producer_work()                                      
  {
        allocate_memory();                      // call a function to allocate memory for matrices
        mat_mat_input();                        // call a function to give input to matrices

  }
 // consumer performs mat-mat multiplication 
 void consumer_work()                       
  {
        mat_mat_multiply();                          // call multiplication function
        
        freememory();           // free memory allocated
  }
 
  // allocate memory
  void allocate_memory()
  {
        //printf("\n nrows = ncols = %d\n",nrows);
        mat1 = (double **)scalable_malloc(sizeof(double) * nrows);
         for (i = 0; i < nrows; i++)
                mat1[i] = (double *)scalable_malloc(sizeof(double) * ncols);


         mat2 = (double **)scalable_malloc(sizeof(double) * nrows);
         for (i = 0; i < nrows; i++)                                                   
                mat2[i] = (double *)scalable_malloc(sizeof(double) * ncols);

         result_mat = (double **) scalable_malloc(sizeof(double) * nrows);
         for (i = 0; i < nrows; i++)
                result_mat[i] = (double *)scalable_malloc(sizeof(double) * ncols);

	printf("\n memory allocated\n");
  }

  // give input to matrices
  void mat_mat_input()
  {    
        for (i=0; i<nrows;i++)
        {
                for (j = 0; j<ncols;j++)
                {
                        mat1[i][j] = (double)(i + j);
                }
        }

        for (i=0; i<nrows;i++)
        {
                for (j = 0; j<ncols;j++)                                     
                {
                        mat2[i][j] = (double)(i + j + 2);
                }
        }

        for (i=0; i<nrows;i++)
        {
                for (j = 0; j<ncols;j++)
                {
                        result_mat[i][j] = 0;
                }
        }

        printf("\n Matrices filled.......\n");

        
  }

  // mat-mat-multiplication
  void mat_mat_multiply()
  {
	for(j=0;j<ncols;j++)
         {
                for(i=0;i<nrows;i++)
                {
                        for(k=0;k<ncols;k++)                                             
                        {
                                result_mat[i][j] = result_mat[i][j] + mat1[i][k] * mat2[k][j];
                        }
                    
                }
        }
        printf("\n matrix multiplication done successfully!!!!\n");
  
  }

 // free memory

  void freememory()
  {     
         scalable_free(mat1);
         scalable_free(mat2);                                                   
         scalable_free(result_mat);
         printf("\n Memory is free now!!!\n");
  }





};

#endif
