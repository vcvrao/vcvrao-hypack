#include"../include/headerfiles.h"
#include"../include/proto.h"

void mmap_matrix_input(char *fname)                                 // give input to matrix
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
          size_t id = j * nrows + i ;
          float val = id;
          nwrite = fwrite (&val, sizeof (float), 1, fp);
          assert (nwrite == 1);
        }
    }
  fclose (fp);
}

