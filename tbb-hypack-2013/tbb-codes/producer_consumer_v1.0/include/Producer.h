
#ifndef __PRODUCER_H__
#define __PRODUCER_H__


#include"../include/headerfiles.h"

using namespace std;
using namespace tbb;

extern list<int>  Q;

typedef spin_mutex MutexType;
extern MutexType mut;

extern int num_of_producers;


struct Producer 
{
  static int index;
 
		
  void operator ()(const blocked_range < size_t > &r) const   // operator function for producer
  {
    int j;
    for (j = r.begin (); j != r.end (); ++j)
      {
		
	MutexType::scoped_lock lock;
	{       
	 lock.acquire(mut);                         // get lock
         Q.push_back(index++);                        //push item in Queue
	 lock.release();			    // release lock	
	}
      }

  }
};


#endif

