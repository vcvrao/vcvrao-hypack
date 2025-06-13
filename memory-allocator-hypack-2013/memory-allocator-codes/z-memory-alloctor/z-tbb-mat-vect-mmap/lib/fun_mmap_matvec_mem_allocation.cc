
#include"../include/headerfiles.h"
#include"../include/define.h"

float* mmap_matvec_mem_allocation(char *fname, int mode, int nrows, int ncols, int *fileid,int flag)
{
                                                                                  // allocate memory using mmap
  int fd, stat;
  size_t mapsize;
  float *map_addr = NULL;

  ncols = nrows;
  int vsize = ncols;
  if(flag == 0)
    mapsize = nrows * ncols * sizeof (float);
  else
    mapsize = vsize * sizeof(float);

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
