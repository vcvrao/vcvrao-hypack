/*****************************************************************************
	

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

	Calculation of PI using Monte-Carlo Method.(UPC)
	TEST_CASES is number of points to be taken by 
        each thread and this can be changed by the user.

Compiling: 
        upcc pie_cal1.upc

Executing:
         upcrun -n 8 ./a.out
	 (Here 8 is the No of threads or No of cores, this can be changed by user)

Result:
	Each Thread Calculates its own value of PI, 
        we can compare the results and take avg.

Created             : August-2013

E-mail              : hpctfe@cdac.in     


**********************************************************************************/

#include<stdio.h>
#include<upc.h>
#include<math.h>
#include<sys/time.h>
#define TEST_CASES 1000000

int calculate_pi()
{
	/* Generate Random Point in 1x1 2-d space */
	double x=(double)rand()/(double)RAND_MAX;
	double y=(double)rand()/(double)RAND_MAX;

	/* Check whether the point belongs to circle of rad 1 unit */
	if(x*x+y*y<=1)return 1;
  	else return 0; 
}


int main()
{
	/* t1,t2 to record start and endtimes of program excution. */
	struct timeval t1,t2;
   	gettimeofday(&t1,NULL);
   	float COUNT=0;
	
	/* Calculate the Pie value By testing Random Points */
   	srand(MYTHREAD*3030);
   	for(long long i=0;i<TEST_CASES;i++)
     		COUNT+=calculate_pi();

	double PI=4*COUNT/(float)TEST_CASES;

   	gettimeofday(&t2,NULL);
  	upc_barrier;

	/* Calculate the execution time i.e t2-t1*/
   	int time=(t2.tv_sec-t1.tv_sec)*1000000+(t2.tv_usec-t1.tv_usec);
   
   	printf("Value of PI by thread %d is %lf....calculated in %lf  seconds\n",MYTHREAD,PI,time/1000000.0);
	upc_barrier;
	upc_global_exit(0);
	return 0;
	
}
