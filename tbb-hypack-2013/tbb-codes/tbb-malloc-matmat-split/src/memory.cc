/* ************************************************ 
	C-DAC Tech Workshop : hyPACK-2013
                October 15-18, 2013

   file : memory.cc
        : tbb_malloc_matmat-split.cc

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

**************************************************** 
*/ 


#include "../include/define.h"
#include "../include/sysheader.h"
#include "../include/proto.h"

/* Function to allocate the memory for matrix using malloc */
float ** map_matrix( float ** mat ,int nrow,int ncol)
{
     int i;
      mat = (float **) malloc(sizeof(float*) * nrow);
                for(i = 0; i < nrow; i++)
                mat[i] = (float *) malloc(sizeof(float) * ncol);
        return mat;

}

/* Function to free memory that is allocated by malloc */

void free_mem(float ** mat,int nrow,int ncol)
{
        int i;
        for( i = 0; i<nrow;i++)
        {
                free(mat[i]);
        }
        free(mat);
}

