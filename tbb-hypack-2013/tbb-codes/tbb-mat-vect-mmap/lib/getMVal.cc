#include"../include/headerfiles.h"

float
getMVal (const float *buf, int i, int j ,int nrows ,int ncols)             // get value from matrix
{
  size_t id = j * nrows + i;
  return buf[id];
}

