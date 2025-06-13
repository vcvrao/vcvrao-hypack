

#include"../include/headerfiles.h"
#include"../include/ParVectVectMult.h"
#include"../include/proto.h"

void par_vector_vector_multiply ()                              // function to perform vec-vec multiplication
{
  ParVectVectMult pmat;
  pmat.vsize = vsize;
  pmat.vectorB = vectorB;
  pmat.vectorA = vectorA;
  pmat.result_vector = result_vector;
  parallel_for (tbb::blocked_range<size_t> (0, vsize), pmat);
  printf("\n Vector-Vector Multiplication done successfully.........\n");
}

