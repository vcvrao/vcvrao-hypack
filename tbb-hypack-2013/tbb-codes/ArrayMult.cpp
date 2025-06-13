/************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example                : ArrayMult.cpp

 Objective              : Multiplication of two array elements and 
                          putting array result in resultant array.

 Demonstrates           : parallel_for().

 Input                  : Length of array

 Output                 : resultant array 

 Created                : August-2013

 E-mail                 : hpcfte@cdac.in     

*************************************************************************/

/////////////////////////////////////////////////////////////////////////////////////////
// Name  : Multiplication of tow array elements and putting array result in resultant array
// Input : Length of array
// Output: resultant array 
/////////////////////////////////////////////////////////////////////////////////////////

#include<iostream>
#include<tbb/task_scheduler_init.h>
#include<tbb/parallel_for.h>
#include<tbb/blocked_range.h>

using namespace std;
using namespace tbb;

/////////////////////////////////////////////////////////////////////////////////////////
//
// This class perform parallel multiplication of element of array
//
/////////////////////////////////////////////////////////////////////////////////////////
class VectMat{
             private:
                double *const vectA, *const vectB, *const resultVect;
             public:
                VectMat(double *a, double *b, double *resultV):vectA(a),vectB(b),resultVect(resultV){} // parameterized constructor
                void operator()(blocked_range<size_t> &r) const
                   { 
                     double *vA, *vB, *vR;
                     vA = vectA;
                     vB = vectB;
                     vR = resultVect;
                     for( size_t count = r.begin(); count != r.end(); count++ )
                        vR[ count ] = vA[count] * vB[count];
                   }


}; //end of class VectMat


int main(int argc, char* argv[])
{
 
  double *vectA, *vectB, *resultVect;
  size_t length;
  
  if( argc == 2) 
   length = atoi(argv[1]);
  else
     {
      cout<<"\n Invalid Number of argument !";
      cout<<"\n Example : ./<executable name> <length of input vector>";
      exit(-1);
     }

  // Allocating memory for input array
  vectA = (double*) malloc(length * sizeof(double));
  vectB = (double*) malloc(length * sizeof(double));
  resultVect = (double*) malloc(length * sizeof(double));

  // initialize input array
  for(size_t count=0; count< length; count++)
     { 
       vectA[count]= vectB[count] = 1.20;
       resultVect[count] = 0.0;
     }   

  // initialize tbb scheduler
  task_scheduler_init init;

  // invoke parallel array element multiplication operation
  parallel_for(blocked_range<size_t>(0,length,1000), VectMat(vectA,vectB,resultVect) );

  //  for(size_t count=0; count< length; count++)
  //       cout<<"  " << resultVect[count];
 



}// end of main

