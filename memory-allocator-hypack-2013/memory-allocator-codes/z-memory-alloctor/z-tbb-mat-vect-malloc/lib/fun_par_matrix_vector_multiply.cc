

#include"../include/headerfiles.h"
#include"../include/ParMatVectMult.h"
//
#include"../include/proto.h"

void par_matrix_vector_multiply ()                   //perform mat-vec multiplication
{
  ParMatVectMult pmat;
  pmat.nrows = nrows;
  pmat.ncols = ncols;
  pmat.vsize = vsize;
  pmat.matrixA = matrixA;
  pmat.vectorA = vectorA;
  pmat.result_vector = result_vector;
  parallel_for (tbb::blocked_range<size_t> (0, nrows), pmat);                  // call operator function
  printf("\n Matrix vector multiplication is done successfully........\n");
}

