* ************************************************ 
	C-DAC Tech Workshop : hyPACK-2013
                October 15-18, 2013

   file : multiply.cc
        : tbb_malloc_matmat-split.cc

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     


**************************************************** 
*/ 

#include "../include/define.h"
#include "../include/sysheader.h"
#include "../include/proto.h"


/* Function to call the call back
   operator function */
void par_matrix_matrix_multiply ()
{
  ParMatrixMult pmat;
  pmat.nrows = nrows;
  pmat.ncols = ncols;
  pmat.ma = ma;
  pmat.mb = mb;
  pmat.mc = mc;
  parallel_for (tbb::blocked_range2d<size_t> (0, nrows,100,0,ncols,100), pmat);
}

