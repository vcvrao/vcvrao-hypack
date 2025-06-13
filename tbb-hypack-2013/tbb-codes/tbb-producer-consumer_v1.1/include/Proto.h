

#ifndef PROTO_H
#define PROTO_H 1

#include"../include/MyWork.h"

using namespace std;
using namespace tbb;


extern concurrent_queue<MyWork> queue;


extern MyWork mm;
extern int num_of_producers;
extern int num_of_consumers; 
extern int size;


#endif
