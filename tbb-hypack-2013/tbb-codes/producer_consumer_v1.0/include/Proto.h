



#include"../include/Producer.h"
#include"../include/Consumer.h"

using namespace std;
using namespace tbb;



int Producer::index = 0;
int Consumer::index = 0;

extern int num_of_producers;
extern int num_of_consumers; 

void producer(int,int);
void consumer(int,int);

