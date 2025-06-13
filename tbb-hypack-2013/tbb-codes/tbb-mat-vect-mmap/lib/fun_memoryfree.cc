

#include"../include/headerfiles.h"

void memoryfree(float *matrixA,float *vectorA,float *result_vector,size_t mapmsize,size_t mapvsize)
{
		munmap (matrixA, mapmsize);
  		munmap (vectorA, mapvsize);
  		munmap (result_vector,mapvsize);
	        printf("\n Memory freed successfully...........\n");

}
