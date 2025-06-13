
/*************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example             : VectorDotOp.cpp

 Objective           : Simple dot operation where a vector element 
                       is multiplied with a scaler value.

 Demonstrates        : parallel_for().

 Input               : Vector length

 Output              : Dot product of vector and scaler multiplication.

 Created             : August-2013

 E-mail              : hpcfte@cdac.in     

****************************************************************/

#include<iostream>
#include "tbb/task_scheduler_init.h"
#include<tbb/parallel_for.h>
#include<tbb/blocked_range.h>
#include<math.h>
using namespace std;
using namespace tbb;

#define MAX_SIZE  200000000

class ApplyFoo{
              private:
                   float *const my_a;
              public:
                   ApplyFoo( float a[]):my_a(a){ }

                   void operator()(const blocked_range<size_t> &r) const
                        {
                          float *a = my_a;
                          for( size_t i = r.begin(); i!=r.end(); ++i)
                             a[i] = rand();
                        }
              };
                         

int main(int argc, char* argv[]) 
{

 int nthread; 
 float *a;
 size_t len = MAX_SIZE;

 
 nthread = strtol(argv[0], 0, 0);
 
 task_scheduler_init init;
 a = (float*)malloc(len* sizeof(float));

   
parallel_for(blocked_range<size_t>(0,len,1000),ApplyFoo(a));
 

 if( nthread >= 1)
   init.terminate();

 return 0;
}// end of main
