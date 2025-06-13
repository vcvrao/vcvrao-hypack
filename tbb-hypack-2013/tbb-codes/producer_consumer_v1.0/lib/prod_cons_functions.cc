
#include"../include/headerfiles.h"
#include"../include/Producer.h"
#include"../include/Consumer.h"


void producer(int num_items,int num_of_producers)                  // producer function definition
{  
    int assign_num,ndiv,nmod;
    ndiv = (num_items / num_of_producers);
    nmod = (num_items % num_of_producers);

    for (int i=0; i<num_of_producers; i++ )
    {
                if(i==(num_of_producers - 1))
			assign_num = ndiv + nmod;
		else
			assign_num = ndiv;
                Producer prod[i];
                printf("\n This Producer %d is producing %d items....",i,assign_num);
                parallel_for (tbb::blocked_range<size_t> (0, assign_num), prod[i]);       // calling operator function for 
    }                                                                               //  producer

}

void consumer(int num_items,int num_of_consumers)             // consumer function definition 
{
         int assign_num,ndiv,nmod;
         ndiv = (num_items / num_of_consumers);
         nmod = (num_items % num_of_consumers);

        for (int i=0; i<num_of_consumers; i++ )
        {
		if(i==(num_of_consumers - 1))
                        assign_num = ndiv + nmod;
                else
                        assign_num = ndiv;

                  
               Consumer cons[i];
               printf("\n This Consumer %d is consuming %d items....",i,assign_num );
               parallel_for (tbb::blocked_range<size_t> (0, assign_num), cons[i]);  // calling operator function for consumer
        }
}
