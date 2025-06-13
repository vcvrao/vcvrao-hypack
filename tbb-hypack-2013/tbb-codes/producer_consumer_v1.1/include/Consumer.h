
#ifndef __CONSUMER_H__
#define __CONSUMER_H__

#include"../include/headerfiles.h"
#include"../include/MyWork.h"

int num_of_consumers; 

struct Consumer
{

  void operator () (const blocked_range < size_t > &r) const           // operator function for consumer
  {
    int j;
    for (j = r.begin (); j != r.end (); ++j)
      {
				
	MutexType::scoped_lock lock;
	{
	  lock.acquire(mut);                                     // get lock 
          if( !Q.empty())                                            //if Q is not empty..pop operation
	  {
	      MyWork w;
              w = Q.front(); Q.pop_front();
              w.consumer_work();

          }
	  lock.release();                                      //release lock
	}
      }

  }
};

#endif
