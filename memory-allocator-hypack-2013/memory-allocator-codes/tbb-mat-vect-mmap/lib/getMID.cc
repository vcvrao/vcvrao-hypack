#include"../include/headerfiles.h"

size_t
getMID (int i, int j ,int nrows , int ncols)              //get matrix location
{
  assert (i >= 0 && i < nrows);
  assert (j >= 0 && j < ncols);

  return j * nrows + i;
}

