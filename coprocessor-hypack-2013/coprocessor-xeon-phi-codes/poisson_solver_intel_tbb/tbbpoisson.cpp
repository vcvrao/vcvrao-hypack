/*****************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example               : tbbposission.cpp

 Objective             : Simple dot operation where a vector element is 
                          multiplied with a scaler value.

 Demonstrates          : parallel_for().

 Input                 : Vector length

 Output                : Dot product of vector and scaler multiplication.

 Created               : August-2013

 E-mail                : hpcfte@cdac.in     

*************************************************************************/
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include<math.h>
#include "assert.h"
#include "tbb/task_scheduler_init.h"
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

using namespace std;
using namespace tbb;

class Poission2D
{
  public:

     static const int LEFT   = 1;
     static const int RIGHT  = 2;
     static const int BOTTOM = 3;
     static const int TOP    = 4;

    Poission2D( float *uo, float *un, int n, int m)
    {  uold = uo; 
       unew = un;
       nx   = n;
       ny   = m;
    }

    void set_boundary_condition(int i, int j, float val)
    {
       int c   = getIndex(i,j);
       uold[c] = val;
       unew[c] = val;
    }

    void set_boundary_condition(int iside, float val) {
	    switch( iside )
	    {
	      case LEFT:
		for( int j = 0; j < ny; j++)
                     set_boundary_condition(0,j, val);
		break;
	      case RIGHT:
		for( int j = 0; j < ny; j++)
                     set_boundary_condition((nx-1),j, val);
		break;
	      case BOTTOM:
		for( int i = 0; i < nx; i++)
                     set_boundary_condition(i,0, val);
		break;
	      case TOP:
		for( int i = 0; i < nx; i++)
                     set_boundary_condition(i,(ny-1), val);
		break;
	    }
    }

    void operator() (const blocked_range<int> &range) const 
    {
       int c, l, r, t, b;
       for(int j = range.begin(); j != range.end(); ++j) {
          for(int i = 1; i < nx-1; i++) {
              c = getIndex(i,j);      // Center node
	      l = getIndex(i-1,j);    // Left node
	      r = getIndex(i+1,j);    // Right node
	      t = getIndex(i,j+1);    // Top node
	      b = getIndex(i,j-1);    // Bottom node
	      unew[c] = 0.25 * ( uold[l] + uold[r] + uold[t] + uold[b] );
          }
     }
    }
  private:
       mutable float *uold, *unew;
       int nx, ny;
       int getIndex( int i, int j) const { return j*nx + i; }

};



class PoissonReduce
{

  public:

   mutable float max_error;


    PoissonReduce( float *uo, float *un, int n, int m)
    {  uold = uo; 
       unew = un;
       nx   = n;
       ny   = m;
       max_error = 0;
    }

    PoissonReduce( PoissonReduce &rhs, split) {
	    uold = rhs.uold;
	    unew = rhs.unew;
	    nx   = rhs.nx;
	    ny   = rhs.ny;
          max_error = 0.0;
    }

    void operator() (const blocked_range<int> &range) const 
    {
       int c;
       float vdiff;

       for(int j = range.begin(); j != range.end(); ++j) {
          for(int i = 1; i < nx-1; i++) {
              c = getIndex(i,j);
	      vdiff = fabs( unew[c]- uold[c]);
	      max_error = std::max( max_error, vdiff );
	      uold[c] = unew[c];
          }
       }
    }

    void join( const PoissonReduce &rhs) {
         max_error = std::max(max_error, rhs.max_error);
    }

   private:
       float *unew, *uold;
       int nx, ny;
       int getIndex( int i, int j) const { return j*nx + i; }
};

//..............................................................//

void Parallel_Poission_Solver( float *u, int nx, int ny, float tolerance, 
    			      int maxIters)
{
   int Index;
   float *unew = (float *) malloc( nx*ny*sizeof(float) );

   Poission2D  poission(u, unew, nx, ny );
   PoissonReduce reduce(u, unew, nx, ny);

   poission.set_boundary_condition(Poission2D::LEFT,   1.0);
   poission.set_boundary_condition(Poission2D::RIGHT,  2.0);
   poission.set_boundary_condition(Poission2D::TOP,    4.0);
   poission.set_boundary_condition(Poission2D::BOTTOM, 3.0);

   int iter = 0;
   while(1) {
        reduce.max_error = 0.0;
	iter++;
       parallel_for( blocked_range<int>(1,ny-1), poission);
       parallel_reduce( blocked_range<int>(1,ny-1), reduce);
       if( (reduce.max_error < tolerance) || (iter == maxIters) ) break;
   }
   printf("The number of iterations is %d \n",iter);
   free( unew );
}

//..............................................................//

int main( int argc, char **argv)
{
    assert( argc == 4 );
    int numx = atoi( argv[1] );
    int numy = atoi( argv[2] );
    int maxIters = atoi( argv[3] );
    float *u;
	
    task_scheduler_init init;

    double tol   = 1.0E-09;

    u = new float[numx*numy];

    for (int i = 0; i < numx * numy; i++) 
        u[i] = 0.0; 

    Parallel_Poission_Solver( u, numx, numy, tol, maxIters );
    for(int Index=0; Index < numx * numy; Index++)
        {
                printf(" \t %d \t  %f \n",Index,u[Index]);
        }

    delete (u);
   
}

