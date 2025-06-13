#include "../include_header/header.h"

	
/*Consumer structure*/
struct Consumer
{
        size_t citems;

        void operator () (const blocked_range < size_t > &r) const
        {
        int j;
 
	for (j = r.begin (); j != r.end (); ++j)
        {

		/* Consuming */
		/* Poppoing an element from the queue*/
		queue.pop(item);
		printf("\n Queue is poped with %d", item);
        }
	}

};


/*Consumer function for number of consumers*/
void consumer(size_t citems)
{

        for ( int i = 0; i < numConsumers; i++ )
        {
                Consumer cons[i];
                cons[i].citems = citems;
       		printf("\t %d \t",TotalProducedItems);
                printf("\n consumer  %d is consuming %d items\n ", i,cons[i].citems);
                parallel_for (tbb::blocked_range<size_t> (0, citems), cons[i]);
                printf("\n consumer  %d is consumed..\n ", i);
        }
}

