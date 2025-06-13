
/* ***********************************************************
*
*		C-DAC Tech Workshop : hyPACK-2013
*                           October 15-18, 2013
*
* File	    : prefix_sum_parallel_scan_autopartitioner.cpp
* Desc      : Computes the prefix sum of a randomly generated 
*             vector upto n-2 terms
* Input     : <vector_size>
* Output    : The prefix sum and the timetaken for the operation 
*             to complete
*
* Created  : August-2013
*
* E-mail   : hpcfte@cdac.in     
*
***************************************************************/

#include<iostream>
#include<tbb/task_scheduler_init.h>
#include<tbb/parallel_scan.h>
#include<tbb/blocked_range.h>
#include<sys/time.h>
#include<stdlib.h>
#include<errno.h>

using namespace std;//standard namespace
using namespace tbb;//tbb namespace

class Body
{
  int sum;//the prefix sum
  int *const y;//y vector
  const int *const x;//x vector

public:
  
  Body (int y_[], const int x_[]):sum (1), x (x_), y (y_)// initialzing sum=1 and x y vectors with the vectors passed
  {
  }

  
  int get_sum () const //wrapper to get sum
  {
    return sum;
  }
  
  template < typename Tag >
  
  void operator () (const blocked_range < int >&r, Tag)
  {
    //actual prefix sum computation in accordance with the grain_size (decided by auto partitioner) 
    int temp = sum;
    for (int i = r.begin (); i < r.end (); ++i)
      {
	temp = temp + x[i];
	if (Tag::is_final_scan ())
	  y[i] = temp;
      }
    sum = temp;
  }

  /*the splitting constructer -
   split b so that this and b can accumulate separately*/
  Body (Body & b, split):x (b.x), y (b.y), sum (1)
  {
  }
  
  /*Merge the preprocessing state of 'a' into 'this' 
    where 'a' was created from 'b' by splitting b's constructor*/
  void reverse_join (Body & a)
  {
    sum = a.sum + sum;
  }
  
  //assign state of b to this
  void assign (Body & b)
  {
    sum = b.sum;
  }

};

//the wrapper fn to start start parallel_scan 
int
DoParallelScan (int y[], const int x[], int n)
{
  Body body (y,x);
  parallel_scan (blocked_range < int >(0, n), body, auto_partitioner());//(grain_size decided by auto_partitioner
  return body.get_sum ();
}

int
main (int argc, char *argv[])
{
  int seed;
  struct timeval tv_start, tv_end; //timeval - one for the start and other for the end
  struct timezone tz_start, tz_end;//timezone - one for start and other for the end
  long timetaken;
  int *vector_x;
  int *vector_y;
  long vector_size;
  size_t grain_size;

  if (argc == 2) // if the user inputs on the vector size
    {
      vector_size = atoi (argv[1]);
    }
  else //if no input by the user
    {
      cout << "assuming vector_size as 1000" << endl;
      vector_size = 1000;
    }

  if (vector_size > 0) //if positive value
  {
       vector_x = new int[vector_size]; // allocating size to the vectors
       vector_y = new int[vector_size];
  }
  else
    perror ("the value of the arrraysize has to be positive\n");
  
  //filling the vectors with randomly generated values
  for (int i = 0; i < vector_size; i++)
    {
      seed = i;
      vector_x[i] =(int) (rand_r ((unsigned int *) &seed) % 8000)+1;
      vector_y[i] =(int) (rand_r ((unsigned int *) &seed) % 8000)+1;
    }
  //initializing tbb task scheduler
  task_scheduler_init init;
  
  //start time
  gettimeofday (&tv_start, &tz_start);

  int result_sum=DoParallelScan(vector_x,vector_y,vector_size-2);

  //end time
  gettimeofday (&tv_end, &tz_end);
  //time taken
  timetaken =
    tv_end.tv_sec * 1000000 + tv_end.tv_usec - (tv_start.tv_sec * 1000000 +
						tv_start.tv_usec);
  cout << "The Prefix sum till n-2 is  :" << result_sum<<endl;//the resultant prefix sum till n-2 
  cout << "Time taken (seconds) :" << timetaken / 1000000 << endl;//converting microseconds into seconds
  cout << "Time taken (u seconds)   :" << timetaken << endl;//time taken in microseconds
  return 0;
}
