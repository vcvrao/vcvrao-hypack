
#ifndef __PRODUCER_H__
#define __PRODUCER_H__


#include"../include/headerfiles.h"
#include"../include/MyWork.h"

std::list<MyWork>  Q;
typedef spin_mutex MutexType;
MutexType mut;

int num_of_producers;

struct Producer 
{
  void operator ()(const blocked_range < size_t > &r) const   // operator function for producer
  {
    int j;
    for (j = r.begin (); j != r.end (); ++j)
    {
		
	MutexType::scoped_lock lock;
	{       
	 	lock.acquire(mut);                 // get lock
         	MyWork mm(50);
         	mm.producer_work();
         	Q.push_back(mm);                   // put work in the workQ
         	lock.release();                    // release lock
	}
     }

  }
};


#endif

