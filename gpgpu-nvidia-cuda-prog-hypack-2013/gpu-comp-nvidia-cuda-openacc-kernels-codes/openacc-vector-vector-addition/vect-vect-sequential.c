
/*
   C-DAC Tech Workshop : hyPACK-2013
        October 15-18, 2013

   Created       : August-2013

   E-mail        : hpcfte@cdac.in     

*/

#include<stdio.h>
#include<sys/time.h>

#define VECSIZE 16384

int main()
{
	float vecA[VECSIZE] ,vecB[VECSIZE],vecRES[VECSIZE];
	int i ;
	double time_start, time_end,memoryused=0.0;
	struct  timeval tv;
	struct timezone tz;

	for(i=0;i<VECSIZE;i++)
	{
		vecA[i]= 0.0;
		vecB[i]= 0.0;
		vecRES[i]= 0.0;
	}

	 gettimeofday(&tv, &tz);
	 time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

	for ( i = 0; i <  VECSIZE ; ++i)
    		vecRES[i] = vecA[i] + vecB[i];


	gettimeofday(&tv, &tz);
   	time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

	printf("\n\t  Sequential Time Vector Addition in  Seconds (T)  :  %lf\n",(time_end - time_start));

	return 0;
}

