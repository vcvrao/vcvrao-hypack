
#include"../include/headerfiles.h"
#include"../include/proto.h"


void vec_memory_allocation(float **vectorB,float **vectorA,float **result_vector) // vector memory allocation
{
	 *vectorB = (float *)malloc(vsize * sizeof(float));
	 *vectorA = (float *)malloc(vsize * sizeof(float));
	 *result_vector	= (float *)malloc(vsize * sizeof(float));
}

