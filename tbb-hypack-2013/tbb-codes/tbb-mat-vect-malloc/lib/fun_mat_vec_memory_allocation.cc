
#include"../include/headerfiles.h"
#include"../include/proto.h"

void mat_vec_memory_allocation(float **matrixA,float **vectorA,float **result_vector)
{
	 *matrixA = (float *)malloc(nrows * ncols * sizeof(float));
	 *vectorA = (float *)malloc(vsize * sizeof(float));
	 *result_vector	= (float *)malloc(nrows * sizeof(float));
}

