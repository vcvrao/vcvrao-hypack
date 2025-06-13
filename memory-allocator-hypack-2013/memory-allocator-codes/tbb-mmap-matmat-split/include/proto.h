/* All Function prototype,inline function ,structure 
   declare/define in this file */

#ifndef _PROTO_H
#define _PROTO_H 1

void fill_matrix (char *fname);

void print_matrix (const float *buf);
float * map_matrix (char *fname, int mode, int nrows, int ncols, int *fileid);
void par_matrix_matrix_multiply ();

using namespace tbb;
inline size_t
getID (int i, int j)
{
  assert (i >= 0 && i < nrows);
  assert (j >= 0 && j < ncols);

  return j * nrows + i;
}

inline float
getVal (const float *buf, int i, int j)
{
  size_t id = getID (i, j);
  return buf[id];
}

inline void
setVal (float *buf, int i, int j, float val)
{
  size_t id = getID (i, j);
  buf[id] = val;
}

struct ParMatrixMult
{
  size_t nrows;
  size_t ncols;
  float *mc, *ma, *mb;

/* call back operator function to parallel
   multiplay the matrix */

  void operator () (const blocked_range2d < size_t > &r) const
  {
    int i, j, k;
    float aik, bkj, sum;
    for (j = r.cols().begin (); j != r.cols().end (); ++j)
      {
        for (i = r.rows().begin(); i!=r.rows().end(); i++)
          {
            sum = 0.0;
            for (k = 0; k < ncols; k++)
              {
                aik = getVal (ma, i, k);
                bkj = getVal (mb, k, j);
                sum += aik * bkj;
              }
            setVal (mc, i, j, sum);
          }
      }
  }
};
#endif
