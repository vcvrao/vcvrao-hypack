

#include"../include/headerfiles.h"

void
setVVal (float *buf, int i, float val ,int nrows)                // set value in a vector
{
  size_t id = i ;
  buf[id] = val;
}

