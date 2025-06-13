

#include"../include/headerfiles.h"

void memoryfree(float *vectorB,float *vectorA,float *result_vector)
{
	free(vectorB);
	free(vectorA);                                                        // freeing memory that is allocated
	free(result_vector);
	printf("\n Memory freed sucessfully..........");
}
