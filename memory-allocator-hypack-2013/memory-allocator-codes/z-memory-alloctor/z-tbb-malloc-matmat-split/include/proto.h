/* All Function prototype,inline function ,structure
   declare/define in this file */

#ifndef _PROTO_H
#define _PROTO_H 1

void fill_matrix(float ** buf);
void print_matrix ( float **buf);
float ** map_matrix( float ** mat ,int nrow,int ncol);
void free_mem(float ** mat,int nrow,int ncol);
void par_matrix_matrix_multiply ();

/* for parallel multiplication structure define here */
using namespace tbb;

struct ParMatrixMult
{
  size_t nrows;
  size_t ncols;
  float **mc, **ma, **mb;

 /* operator function start here
   that  is call back function to
   perform matrix multiplication
   in parallel */

  void operator () (const blocked_range2d < size_t > &r) const
  {
    int i, j, k;
    float aik, bkj, sum;
    for (i = r.rows().begin (); i!= r.rows().end (); ++i)
      {
        for (j = r.cols().begin(); j!=r.cols().end(); j++)
          {
            sum = 0.0;
            for (k = 0; k < ncols; k++)
              {
                sum += ma[j][k]*mb[k][i];
              }
             mc[i][j] = sum;
          }
      }
  }
};

#endif
