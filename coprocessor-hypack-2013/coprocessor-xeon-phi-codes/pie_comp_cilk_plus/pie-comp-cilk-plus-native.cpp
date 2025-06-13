/**********************************************************************
 *                    C-DAC Tech Workshop : hyPACK-2013
 *                          October 15-18,2013
 *Example 1     : pie-comp-cilk-plus-native.cpp
 *Objective     : Write An Cilk Plus Program For Pie Calculation
 *Input         : 1)Number of Iterations
 *                2)Number of threads
 *Output        : 1)Time taken in seconds
 *Created       : August-2013    
 *E-Mail        : Betatest@Cdac.In                                          
 ***************************************************************************/

#include <iostream>
#include <iomanip>
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<cilk/cilk.h>
#include<cilk/cilk_api.h>
#include<sys/time.h>

using namespace std;

double walltime()
{
	double tsec=0.0;
	struct timeval mytime;
	gettimeofday(&mytime,0);
	tsec=(double)(mytime.tv_sec+mytime.tv_usec*1.0e-6);
	return tsec;
}


int main(int argc,char* argv[]) 
{
	
	double start_time,end_time,timetaken;
	unsigned int decP=atoi(argv[1]);
	unsigned int denom=3;
	float ourPi=4.0f;
	bool addFlop=true;
	
	__cilkrts_set_param("nworkers", argv[2]);	

	start_time=walltime();	
	_Cilk_for (unsigned int i=1;i<=decP;i++) 
	{
		if (addFlop) 
		{
			ourPi-=(4.0/denom);
			addFlop=false;
			denom+=2;
		}
		else 
		{
			ourPi+=(4.0/denom);
			addFlop=true;
			denom+=2;
		}
	}
	end_time=walltime();
	timetaken=end_time-start_time;

	cout << "Pi calculated with " << decP << " iterations is: ";
	cout << ourPi << endl;
	cout<<"Timetaken:"<<timetaken<<endl;

	return 0;
}
