
/* ************************************************ 
	C-DAC Tech Workshop : hyPACK-2013
                October 15-18, 2013

   file : input.cc
        : tbb_mmap_matmat-split.cc

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     


**************************************************** 
*/ 

#include "../include/define.h"
#include "../include/sysheader.h"
#include "../include/proto.h"

/*Function to prepare input for the matrix */

void fill_matrix (char *fname)
{
  FILE *fp = fopen (fname, "w+");
  if (fp == NULL)
    {
      printf (" Cann't open the file: %s \n", fname);
      return;
    }

  int counter = 1;
  int nwrite;
  for (size_t j = 0; j < ncols; j++)
    {
      for (size_t i = 0; i < nrows; i++)
        {
          size_t id = getID (i, j);
          float val = id;
          nwrite = fwrite (&val, sizeof (float), 1, fp);
          assert (nwrite == 1);
        }
    }
  fclose (fp);
}

