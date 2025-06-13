
#include"../include/headerfiles.h"
#include"../include/define.h"

float* mmap_vec_mem_allocation(char *fname, int mode, int vsize, int *fileid)
{
                                                                                  // allocate memory using mmap
  int fd, stat;
  size_t mapsize;
  float *map_addr = NULL;

  mapsize = vsize * sizeof(float);

  if (mode == MAP_RDONLY)
    fd = open (fname, O_RDONLY);

  if (mode == MAP_RDWR)
    fd = open (fname, O_RDWR);

  if (fd <= 0)
    {
      printf ("Error: Cann't open file  \n");
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
