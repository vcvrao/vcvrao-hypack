#
#
#	C-DAC Tech Workshop : hyPACK-2013
#                  October 15-18, 2013
#
#
#   Created       : August-2013
#
#   E-mail        : hpcfte@cdac.in     
#
#
#include<openacc.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"timer.h"
//const int size = 10240;
#define EPS 1.0e-8f
void fill_sp_matrix(float* matrix,int rowSize,int colSize)
{
        int     row, col ;

        for( row=0; row < rowSize; row++)
             for( col=0; col < colSize; col++)
                        matrix[row * colSize + col] = rand()%10;

}


void MatMatMult(float * restrict a,float * restrict b, float * restrict c, int m, int n, int p) 
{ 
    int i, j, k ; 

StartTimer();
#pragma acc data region copyin(a[0:(m*n)-1]), copyin(b[0:(m*p)-1]),copyout(c[0:(p*n)-1]) 
{ 
	#pragma acc kernels loop gang, vector(16)
    	for (i=0; i<m; i++)
	{ 
		#pragma acc loop gang, vector (16) 
        	for (j=0; j<n; j++) 
        	{ 
            		float sum = 0.0 ; 
			#pragma acc for seq 
            		for (k=0; k<p; k++) 
                		sum += a[i*p+k]*b[k*n+j] ; 
            		c[i*n+j] = sum ; 
        	} 
    	} 
} 
double runtime = GetTimer();
printf("\nOpenACC time in sec= %f\n", runtime/1000);
}

int checkResult(int size, float *hMatC_gpu, float *hMatA, float *hMatB)
{
	float sum;
        float  errorNorm = 0.0;
        float  eps=EPS;
        float  relativeError=0.0;
        int     flag=0;
	int i,j,k,step=0;
	float *hMatC_seq;
	hMatC_seq = (float *) malloc(size*size*sizeof(float));
	for(i=0;i<size*size;i++)
        {
                hMatC_seq[i]=0;
        }	
    StartTimer();
     for ( i = 0; i < size; ++i) {
        for ( j = 0; j < size; ++j) {
		sum = 0.0;
            for ( k = 0; k < size; ++k) {
                sum += hMatA[i*size+k]*hMatB[k*size+j] ;
            }
		hMatC_seq[i*size+j] = sum ;
        }
	}
	double runtime = GetTimer();
    	printf("\nserial time in sec= %f\n", runtime/1000);
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
		{
			if(fabs(hMatC_seq[i*size+j]) > fabs(hMatC_gpu[i*size+j]))
			{
				relativeError = fabs(((hMatC_seq[i*size+j]) - (hMatC_gpu[i*size+j])) / (hMatC_seq[i*size+j]));
			}
			else
			{
				relativeError = fabs(((hMatC_gpu[i*size+j]) - (hMatC_seq[i*size+j])) / (hMatC_gpu[i*size+j]));			  
			}
			if(relativeError > eps && relativeError != 0.0e+00)
			{
				if(errorNorm < relativeError)
				{
					errorNorm = relativeError;
					//printf("failed");
					flag = 1;
				}
			}
			else
			{
				flag = 0;
				//printf("success");
			}
		}
	}
	if(flag == 1)
	{
		printf("\nResult verification  : failed\n");
		printf("\nRelative error = %e\n", errorNorm);
	}
	else
	{
		printf("\nResult verification : success\n");
	}
	free(hMatC_seq);	
} 
int main(int argc, char *argv[])
{
	int i;
	float *restrict hMatA;
	float sum =0.0;
	float *restrict hMatB;
	float *restrict hMatC;
	float *hMatC_seq;
	int size;
	if(argc < 2)
	{
		 size = 1024;
	}
	else
	{
		size = atoi(argv[1]);
	}
	printf("\n matrix size : %d\n", size);
	hMatA = (float *) malloc(size*size*sizeof(float));
	hMatB = (float *) malloc(size*size*sizeof(float));
	hMatC = (float *) malloc(size*size*sizeof(float));

	// Initialize buffers.
   	fill_sp_matrix(hMatA,size,size); 
   	fill_sp_matrix(hMatB,size,size); 
	for(i=0;i<size*size;i++)
        {
                hMatC[i]=0;
        }	

    	acc_init( acc_device_nvidia ); //connect host to device

  	MatMatMult(hMatA,hMatB,hMatC,size,size,size);

	checkResult(size, hMatC, hMatA, hMatB);   
    return 0;
}
