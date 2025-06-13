
#ifndef __CONSUMER_H__
#define __CONSUMER_H__

#include"../include/headerfiles.h"

using namespace std;
using namespace tbb;

extern int num_of_consumers; //= 20 ;

struct Consumer
{
  static int index;

  void operator () (const blocked_range < size_t > &r) const           // operator function for consumer
  {
    int j;
    for (j = r.begin (); j != r.end (); ++j)
      {
				
	MutexType::scoped_lock lock;
	{
	  lock.acquire(mut);                                     // get lock 
          if( !Q.empty())                                            //if Q is not empty..pop an item
	  {
	      int item  = Q.front(); Q.pop_front();
              index++;
              //printf("\n Item popped %d\n",item);
          }
	  lock.release();                                      //release lock
	}
      }

  }
};

#endif
