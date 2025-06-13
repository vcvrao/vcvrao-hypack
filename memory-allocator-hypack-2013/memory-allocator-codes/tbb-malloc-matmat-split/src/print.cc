/* ************************************************ 
	C-DAC Tech Workshop : hyPACK-2013
                October 15-18, 2013

   file : print.cc
        : tbb_mmalloc_matmat-split.cc

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     


**************************************************** 
*/ 

#include "../include/define.h"
#include "../include/sysheader.h"
#include "../include/proto.h"

/* Function to print the matrix */

void print_matrix ( float **buf)
{
  float val;
  for (size_t i = 0; i < nrows; i++)
    {
      for (size_t j = 0; j < ncols; j++)
        {
          printf ("%lf ", buf[i][j]);
        }
      printf ("\n \n");
    }
}

