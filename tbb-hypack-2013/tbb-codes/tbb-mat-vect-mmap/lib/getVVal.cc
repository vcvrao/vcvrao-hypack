

#include"../include/headerfiles.h"

float
getVVal (const float *buf, int i,int nrows)                             // get value from vector
{
  size_t id = i;
  return buf[id];
}

