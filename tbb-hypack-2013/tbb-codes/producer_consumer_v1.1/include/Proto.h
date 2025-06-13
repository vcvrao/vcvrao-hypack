


#ifndef PROTO_H
#define PROTO_H 1

#include"../include/MyWork.h"

using namespace std;
using namespace tbb;


extern std::list<MyWork>  Q;

typedef spin_mutex MutexType;
extern MutexType mut;

extern int num_of_producers;
extern int num_of_consumers; 
extern int size;


#endif
