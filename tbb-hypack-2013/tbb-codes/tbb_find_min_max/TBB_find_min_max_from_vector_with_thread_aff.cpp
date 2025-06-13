/*
**************************************************************
 *
 *	C-DAC Tech Workshop : hyPACK-2013
 *             October 15-18, 2013
 *
 * Program: To determine the minimum and maximum values 
 * in a vector (array).
 * 
 * Input  : Array/Vector size , Grain Size (optional)
 * Output : Minimum Value, MinVal index, Maximum Value, MaxVal index, 
 * 		Time taken (in sec), Time taken (in microsec). 
 *
 *  Created     : August-2013
 *
 *  E-mail      : hpcfte@cdac.in     

**************************************************************
 */


#include <iostream>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <float.h>//for FLT_MAX
#include <sched.h>//for sched_getaffinity()
#include <sys/time.h>
#include <stdlib.h>
#include <errno.h>

using namespace std;
using namespace tbb;

class MinMaxCalc
{
private:
	const float *const ArrayA;
public:
	float MinValue ;//minimum value 
	long IndexMinVal ;//index of the minimum value of the vector
	float MaxValue ;//maximum value 
	long IndexMaxVal ;//index of the maximum value of the vector
	
	void operator()( const blocked_range<size_t>& r)
	{
		const float *a = ArrayA;
		float val;
		for(size_t i=r.begin(); i!=r.end(); ++i)
		{
			val = a[i];
			if(val < MinValue)//if less than MinValue, update MinValue
			{
				MinValue = val;
				IndexMinVal = i;
			}
			if(val > MaxValue)//if greater than MaxValue, update MaxValue
			{
				MaxValue = val;
				IndexMaxVal = i;
			}
		}
	}
	
	MinMaxCalc(MinMaxCalc &x, split):
		ArrayA(x.ArrayA), 
		MinValue(FLT_MAX), MaxValue(-FLT_MAX),
		IndexMinVal(-1), IndexMaxVal(-1) {}
		
	
	void join(const MinMaxCalc &y)
	{
		if(y.MinValue < MinValue)//if MinValue passed to join is less than current MinValue
		{
			MinValue = y.MinValue;
			IndexMinVal = y.IndexMinVal;
		}
		if(y.MaxValue > MaxValue)//if MaxValue passed to join is greater than current MaxValue
		{
			MaxValue = y.MaxValue;
			IndexMaxVal = y.IndexMaxVal;
		}
	}
	
	MinMaxCalc(const float * A):
		ArrayA(A),//initialize ArrayA to A 
		MinValue(FLT_MAX),//initialize MinValue to FLT_MAX
		IndexMinVal(-1),//initialize IndexMinVal to -1
		MaxValue(-FLT_MAX),//initialize MaxValue to FLT_MAX 
		IndexMaxVal(-1) //initialize IndexMaxVal to -1
		{}
};


int main(int argc, char * argv[])
{
	int rc;//return code
	int seed;//seed for rand_r()
	struct timeval tv_start, tv_end;//for gettimeofday()
	struct timezone tz_start, tz_end;//for gettimeofday()
	long timetaken;//for gettimeofday()
	
	unsigned long ThreadAffMask;//to obtain Thread Affinity (no. of processors to which thread is bound)
	unsigned int ThreadAffMaskLen = sizeof(ThreadAffMask);//length of AffinityMask
	pid_t ProgPid = getpid();//PID of the program for thread affinity 
		
	float * Array;
	long ArraySize;
	size_t GrainSize;
	if(argc == 4)//Thread Affinity is given
	{
		ArraySize = atoi(argv[1]);
		GrainSize = atoi(argv[2]);
		if(atoi(argv[3])>0)
			ThreadAffMask = atoi(argv[3]);
		rc = sched_setaffinity(ProgPid, ThreadAffMaskLen, (cpu_set_t *)&ThreadAffMask);
		if(rc == -1)
			cout << "Couldn't set thread affinity\n";
	}
	else if(argc == 3)//Array size & GrainSize are specified.
	{
		ArraySize = atoi(argv[1]);
		GrainSize = atoi(argv[2]);
	}
	else if(argc == 2)//ArraySize is given
	{
		cout << "usage : " << argv[0] << " <ArraySize> <GrainSize>\n"
				"assuming  GrainSize = 10000\n";
		ArraySize = atoi(argv[1]);
		GrainSize = 10000;//as default value
	}
	else
	{
		cout << "usage : " << argv[0] << " <ArraySize> <GrainSize> <ThreadAffinity-Mask>\n"
				"assuming ArraySize = 1000\n"
				"         GrainSize = 10000\n";
		ArraySize = 1000;//as default value
		GrainSize = 10000;//as default value
	}
	
	if(ArraySize > 0)//if positive value
		Array = new float[ArraySize];
	else
		perror("Value of ArraySize should be > 0\n");
	
	for(int i=0; i<ArraySize; i++)//assign random values to Array
	{
		seed = i;
		Array[i] = ( rand_r((unsigned int*)&seed) % 2000 ) + 1.5;//any value between 0 and ArraySize
	}
	
	//initialize intel TBB task scheduler
	task_scheduler_init init;
	
	MinMaxCalc MMC(Array);//create & initialize object of MinMaxCalc
	
	//start time
	gettimeofday(&tv_start, &tz_start);
	
	//call parallel_reduce for parallel computation of MinValue & MaxValue
	parallel_reduce(blocked_range<size_t>(0,ArraySize,GrainSize), MMC);
	
	//end time
	gettimeofday(&tv_end, &tz_end);
	//time taken
	timetaken = tv_end.tv_sec*1000000 + tv_end.tv_usec - (tv_start.tv_sec*1000000 + tv_start.tv_usec);
	
	//get thread affinity
	if(argc == 4)//if thread affinity has been given
	{
		rc = sched_getaffinity(ProgPid, ThreadAffMaskLen, (cpu_set_t *)&ThreadAffMask);
		printf(" pid %d's executed with affinity: %08lx\n\n", ProgPid, ThreadAffMask);
	}
	cout << "Results:\n";
	cout << "-----------------------------------------------------------------------------------------\n";
	cout << "Minimum Value of the vector = " << MMC.MinValue << " , @ index = " << MMC.IndexMinVal << "\n"
			"Maximum Value of the vector = " << MMC.MaxValue << " , @ index = " << MMC.IndexMaxVal << "\n"
			"Time taken (seconds)        = " << timetaken / 1000000 << "\n"
			"Time taken (micro seconds)  = " << timetaken << "\n"
			"-----------------------------------------------------------------------------------------\n";
		
	
	return 0;
}
