#include"../include/headerfiles.h"
#include"../include/proto.h"

void matrix_vector_input(float *matrixA , float *vectorA)                    
{

	int i,j;
	for(i=0;i<nrows;i++)
	{
		for(j=0;j<ncols;j++)
		{
			size_t id = j * nrows + i ;
			float val = id;
			matrixA[id]= val;                             // give input to matrix
		}
	}

	
        for(i=0;i<nrows;i++)
	{
		size_t vid = i;
		float vval = vid;                                     // give input to vector
		vectorA[i] = vval;
	}

}
