/**********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013


 Example                : ParellelReduceaMinMax.cpp

 Objective              : Advance example program, to do parallel 
                          multiple reduction operation at same 
                          time and finding min and max value with in an array.

 Demonstrates           : parallel_for().

 Input                  : Array of integer 

 Output                 : result of +reduction, *reduction and Max and Min element 
                          within input array     

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     


************************************************************************/

#include<iostream>
#include<tbb/task_scheduler_init.h>
#include<tbb/blocked_range.h>
#include<tbb/parallel_reduce.h>
#include<tbb/tick_count.h>
using namespace std;
using namespace tbb;

//////////////////////////////////////////////////////////////////////////////////////////
//
// This class uses intel tbb parallel reduce function to perform certain operation
//
//////////////////////////////////////////////////////////////////////////////////////////
template<class T>
class ParallelReduction{
                       private:
                          T *inputArray;
                          T sum, mult;
                          T max, min;
                       public:
                          ParallelReduction(){ inputArray = NULL; sum  = mult = max = min = 0; }
                          ParallelReduction(T *iArray):inputArray(iArray),sum(0),mult(0),max(0),min(0){ }
                          ParallelReduction(ParallelReduction& pReduction):inputArray(pReduction.inputArray),sum(0), mult(0), max(0), min(0){ }
                          ParallelReduction(ParallelReduction& pReduction, split):inputArray(pReduction.inputArray),sum(0), mult(0), max(0), min(0){}
                          T getSum(){ return sum; }
                          T getMult(){ return mult; }
                          T getMax(){ return max; }
                          T getMin(){ return min; }
                          void operator()(const blocked_range<size_t> &r)
                            {
                               for(size_t count = r.begin(); count != r.end(); count++) 
                                  {
                                    sum += inputArray[count];
                                    mult *= inputArray[count];
                                    max = ( max > inputArray[count]? max: inputArray[count]);
                                    min = ( min > inputArray[count]? inputArray[count]:min);
                                  }
//                               cout<<"\n sub sum : "<<sum;
                            }
 
                          void join(const ParallelReduction& pReductionSub)
                            {
                               sum  += pReductionSub.sum; 
                               mult += pReductionSub.mult; 
                               max   = (max > pReductionSub.max?max : pReductionSub.max);  
                               min   = (min > pReductionSub.min?pReductionSub.min : min);  
                            }


};// end of class ParallelReduction

//////////////////////////////////////////////////////////////////////////////////////////
//
// This class uses simple serial way to perform operation
//
//////////////////////////////////////////////////////////////////////////////////////////
template<class T>
class SerialReduction{
                      private: 
                          T *inputArray;
                          T sum, mult;
                          T max, min;
                      public:
                          SerialReduction(){ inputArray = NULL; sum  = mult = max = min = 0; }
                          SerialReduction(T *iArray):inputArray(iArray),sum(0),mult(0),max(0),min(0){ }
                          SerialReduction(SerialReduction& sReduction):inputArray(sReduction.inputArray),sum(0), mult(0), max(0), min(0){ }
                          T getSum(){ return sum; }
                          T getMult(){ return mult; }
                          T getMax(){ return max; }
                          T getMin(){ return min; }
                          void reductionOperation(size_t length)
                           { 
                               for(size_t count = 0; count <= length ; count++) 
                                  {
                                    sum += inputArray[count];
                                    mult *= inputArray[count];
                                    max = ( max > inputArray[count]? max: inputArray[count]);
                                    min = ( min > inputArray[count]? inputArray[count]:min);
                                  }
                            }

                     }; // end of class SerialReduction


//////////////////////////////////////////////////////////////////////////////////////////
// NOTE : The following define statement is specifying data type of input array,
//        on which you want to perform operation [ data type can be like : int, double, float etc. ]
//////////////////////////////////////////////////////////////////////////////////////////

#define DATA_TYPE double               

//////////////////////////////////////////////////////////////////////////////////////////

int main( int argc, char* argv[])
 {

  DATA_TYPE *inputArray;
  size_t length = 0;

  if( argc == 2)
      length = atoi(argv[1]);
  else
      {
       cout<<"\n Invalid Number of argument !";
       cout<<"\n Example : ./<executable name> <length of input vector>\n";
       exit(-1);
      }

  // allocating memory to input array
  inputArray = (DATA_TYPE*)malloc( length *sizeof(DATA_TYPE) );
  for( size_t count=0; count < length; count++)
     inputArray[count] = count ;
 
  // Serial reduction section
  cout<<"\n-------------------Serial Section-------------------------------------\n";
  SerialReduction<DATA_TYPE> sr(inputArray); 
  
  tick_count sStart = tick_count::now();
  sr.reductionOperation(length);
  tick_count sEnd = tick_count::now();
  
  cout<<" Time Taken : "<< (sEnd - sStart).seconds();
  cout<<"\n +Reduction : "<< sr.getSum();
  cout<<"\n *Reduction : "<< sr.getMult();
  cout<<"\n Max Value  : "<< sr.getMax();
  cout<<"\n Min Value  : "<< sr.getMin();


  // Parallel reduction section
  cout<<"\n-------------------Parallel Section-------------------------------------\n";
  ParallelReduction<DATA_TYPE> pr(inputArray);
  task_scheduler_init init;
  
  tick_count pStart = tick_count::now();
  parallel_reduce( blocked_range<size_t>(0,length,10000), pr );
  tick_count pEnd = tick_count::now();
  
  cout<<" Time Taken : "<< (pEnd - pStart).seconds();
  cout<<"\n +Reduction : "<< pr.getSum();
  cout<<"\n *Reduction : "<< pr.getMult();
  cout<<"\n Max Value  : "<< pr.getMax();
  cout<<"\n Min Value  : "<< pr.getMin();

  cout<<"\n\n";
  return 0;

 }// end of main

