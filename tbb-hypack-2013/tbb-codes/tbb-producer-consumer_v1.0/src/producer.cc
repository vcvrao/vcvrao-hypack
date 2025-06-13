#include "../include_header/header.h"

using namespace tbb;
using namespace std;

int numItems, TotalProducedItems, numProducers, numConsumers, count;
concurrent_queue<int> queue;
static int item = 0;
size_t items;


/*Producer Structure*/
struct Producer
{
        size_t items;

        void operator () (const blocked_range < size_t > &r) const
        {
        int j;
        for (j = r.begin (); j != r.end (); ++j)
        {
		/*producing*/
		/*Pushing an element in to the queue*/
		queue.push(item);
		printf("\n Queue is filled with %d", item);
		item++;
        }
        }

};


/*Producer function for number of consumers*/
void producer(size_t items)
{
        for ( int i = 0; i < numProducers; i++ )
        {
                Producer prod[i];
                prod[i].items = items;
                printf("\n producer  %d is producing %d items\n ", i,prod[i].items);
                parallel_for (tbb::blocked_range<size_t> (0, items), prod[i]);
                printf("\n producer  %d is produced..\n ", i);
        }
}

