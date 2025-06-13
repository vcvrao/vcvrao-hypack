
/* ************************************************ 
	C-DAC Tech Workshop : hyPACK-2013
                October 15-18, 2013
   file : memory.cc
        : tbb_mmap_matmat-split.cc

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     


**************************************************** 
*/ 

#include "../include/define.h"
#include "../include/sysheader.h"
#include "../include/proto.h"

/* Function to allocate the memory for matrics
   using mmap memeory allocator */

float *
map_matrix (char *fname, int mode, int nrows, int ncols, int *fileid)
{
  int fd, stat;
  size_t mapsize;
  float *map_addr = NULL;

  ncols = nrows;
  mapsize = nrows * ncols * sizeof (float);

  if (mode == MAP_RDONLY)
    fd = open (fname, O_RDONLY);

  if (mode == MAP_RDWR)
    fd = open (fname, O_RDWR);

  if (fd <= 0)
    {
      printf ("Error: Cann't open file matrixA \n");
      return NULL;
    }

  if (mode == MAP_RDONLY)
    map_addr = (float *) mmap (0, mapsize, PROT_READ, MAP_SHARED, fd, 0);

  if (mode == MAP_RDWR)
    map_addr =
      (float *) mmap (0, mapsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

  if (map_addr == MAP_FAILED)
    {
      printf ("Error: mmap failed \n");
      exit (0);
      return NULL;
    }
  *fileid = fd;
 return map_addr;
}

