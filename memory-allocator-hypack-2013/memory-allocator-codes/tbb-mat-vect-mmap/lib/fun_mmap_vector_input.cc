
#include"../include/headerfiles.h"
#include"../include/proto.h"

void mmap_vector_input(char *fname)                               // give input to vector
{
  FILE *fp = fopen (fname, "w+");
  if (fp == NULL)
    {
      printf (" Cann't open the file: %s \n", fname);
      return;
    }

  int counter = 1;
  int nwrite;
      for (size_t i = 0; i <vsize; i++)
        {
          size_t id = i;
          float val = id;
          nwrite = fwrite (&val, sizeof (float), 1, fp);
          assert (nwrite == 1);
        }

  fclose (fp);
}

