/*
   C-DAC Tech Workshop : hyPACK-2013
         October 15-18, 2013

   Prog. : Vector Vectir Addition using OpenACC 

   Created   : August-2013

   E-mail    : hpcfte@cdac.in     

*/

#include<stdio.h>
#include<sys/time.h>
#include <openacc.h>

#define VECSIZE 16384

int main()
{
	float vecA[VECSIZE] ,vecB[VECSIZE],vecRES[VECSIZE];
	int i ;
	double time_start, time_end,memoryused=0.0;
	struct  timeval tv;
	struct timezone tz;
#if _OPENACC
       acc_init(acc_device_nvidia);
#endif

	for(i=0;i<VECSIZE;i++)
	{
		vecA[i]= 0.0;
		vecB[i]= 0.0;
		vecRES[i]= 0.0;
	}

	 gettimeofday(&tv, &tz);
	 time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

	
	#pragma acc parallel 
	#pragma acc kernels loop gang(32),vector(16)
	for ( i = 0; i <  VECSIZE ; ++i)
    		vecRES[i] = vecA[i] + vecB[i];


	gettimeofday(&tv, &tz);
   	time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

	printf("\n\t OpenACC Vector Addition Step1 Time in  Seconds (T)  :  %lf\n",(time_end - time_start));

	return 0;
}

