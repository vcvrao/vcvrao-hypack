#include"../includeheaderfiles.h"

void
setMVal (float *buf, int i, int j, float val,int nrows ,int ncols)          // set a value in matrix at particular location
{
  size_t id =  j * nrows + i;
  buf[id] = val;
}

