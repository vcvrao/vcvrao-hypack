

#include"../include/headerfiles.h"

void memoryfree(float *matrixA,float *vectorA,float *result_vector)
{
	free(matrixA);
	free(vectorA);                                                        // freeing memory that is allocated
	free(result_vector);
	printf("\n Memory freed sucessfully..........");
}
