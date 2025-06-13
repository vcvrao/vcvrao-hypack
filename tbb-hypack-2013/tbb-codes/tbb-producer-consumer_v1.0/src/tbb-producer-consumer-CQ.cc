
/*******************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

        Example       : tbb-producer-consumer-CQ.cc

        Objective     : To explain the producer consumer problem with 
                        concurrent queue using Intel TBB 
     
        Input         : No of Items, No of Producers, No of Consumers

        Output        : Sequence of Produced and Consumed items

       Created        : August-2013

       E-mail         : hpcfte@cdac.in     

***********************************************************************/

#include "../include_header/header.h"
#include "producer.cc"
#include "consumer.cc"


int main(int argc, char *argv[])
{
        task_scheduler_init init;

	if (argc != 4){
	printf("\n\t Very Few Arguments\n ");
	printf("\n\t Syntax : exec <numItems> <numProducers> <numConsumers> \n");
	printf("\n\t Aborting...!!!\n");
	return 0; 
	}

	else{
	numItems = atoi(argv[1]);
	numProducers = atoi(argv[2]);
	numConsumers = atoi(argv[3]);
	}

        printf("\n Producer is producing goods \n");
        producer(numItems);
        printf("\n Producer has produced goods \n");
	
	TotalProducedItems=item;	
	printf("\n Total Produced goods are %d", TotalProducedItems);        
        printf("\n Consumer is consuming goods \n");
        consumer(numItems);
        printf("\n Consumer has consumed goods \n");

        return 0;
}
  
