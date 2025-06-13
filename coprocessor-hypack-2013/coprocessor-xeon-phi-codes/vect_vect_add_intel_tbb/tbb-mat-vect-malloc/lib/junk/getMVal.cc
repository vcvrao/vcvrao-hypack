

#include"../include/headerfiles.h"

float
getMVal (const float *buf, int i, int j ,int nrows ,int ncols)
{
  size_t id = j * nrows + i;                                           // get value from matrix
  return buf[id];
}

