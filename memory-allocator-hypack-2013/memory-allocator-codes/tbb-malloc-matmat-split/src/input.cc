/* ************************************************ 
	C-DAC Tech Workshop : hyPACK-2013
                October 15-18, 2013

   file : input.cc
        : tbb_malloc_matmat-split.cc

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     


**************************************************** 
*/ 

#include "../include/define.h"
#include "../include/sysheader.h"
#include "../include/proto.h"
/* Function to assign value in the matrix*/
void fill_matrix(float ** buf)
{
    for (size_t i = 0; i < nrows; i++)
    {
      for (size_t j = 0; j <ncols; j++)
        {

                buf[i][j] = i+j;
        }
    }

}

