#include"../include/headerfiles.h"
#include"../include/proto.h"

void vector_input(float *vectorB , float *vectorA)                    
{

	int i;
	for(i=0;i<vsize;i++)
	{
			size_t id =  i ;
			float val = id;
			vectorB[i]= val;                             // give input to vectorB
	}

	
        for(i=0;i<vsize;i++)
	{
		size_t vid = i;
		float vval = vid;                                     // give input to vectorA
		vectorA[i] = vval;
	}

}
