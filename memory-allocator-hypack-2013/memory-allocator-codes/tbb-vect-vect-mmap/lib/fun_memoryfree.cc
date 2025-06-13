

#include"../include/headerfiles.h"

void memoryfree(float *vectorB,float *vectorA,float *result_vector,size_t mapvsize)   // function to free memory
{
		munmap (vectorB, mapvsize);
  		munmap (vectorA, mapvsize);
  		munmap (result_vector,mapvsize);
	        printf("\n Memory freed successfully...........\n");

}
