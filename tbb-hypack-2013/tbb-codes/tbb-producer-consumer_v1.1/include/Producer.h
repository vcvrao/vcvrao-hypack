
#ifndef __PRODUCER_H__
#define __PRODUCER_H__


#include"../include/headerfiles.h"
#include"../include/MyWork.h"

concurrent_queue<MyWork> queue;

int num_of_producers;

struct Producer 
{
  void operator ()(const blocked_range < size_t > &r) const   // operator function for producer
  {
    int j;
    for (j = r.begin (); j != r.end (); ++j)
    {
		
         	MyWork mm(50);
         	mm.producer_work();
		queue.push(mm);
     }


  }
};


#endif

