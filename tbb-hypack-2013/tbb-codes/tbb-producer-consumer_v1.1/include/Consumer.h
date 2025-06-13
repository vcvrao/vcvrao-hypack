
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
				
	      MyWork mm;
              queue.pop(mm); 
              mm.consumer_work();

      }

  }
};

#endif
