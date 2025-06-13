
/*************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

        Example       : producer_consumer_v1.0.cc

        Objective     : To explain the producer consumer problem using Intel TBB

        Input         : Number of threads, Number of Items, 
                        Number of Producers, Number of Consumers

        Output        : Sequence of Produced and Consumed items

        Created       : August-2013

        E-mail        : hpcfte@cdac.in     

*************************************************************************/

#include"../include/headerfiles.h"            // including header files
#include"../include/Proto.h"

list<int>  Q;
int num_of_producers;                                     
int num_of_consumers;                        // variable declarations              
MutexType mut;

int main (int argc,char **argv) 
{
  int num_items;
  int num_of_producers,num_of_consumers;
  int numThreads;


   if (argc != 5)
   {
      printf ("Usage: executable # numThreads #numItems #numProducers #numConsumers \n");
      return 1;
   }

  numThreads =  atoi (argv[1]);
  num_items =  atoi (argv[2]);
  num_of_producers =  atoi (argv[3]);
  num_of_consumers =  atoi (argv[4]);

  tbb::task_scheduler_init init (numThreads);
  

   printf("\n---------------------------------------Producer Work starts here----------------------------------------\n");

   producer(num_items,num_of_producers);                      // call producer

   printf("\n---------------------------------------Producer Work ends here----------------------------------------\n");

   printf("\n---------------------------------------Consumer Work starts here----------------------------------------\n");

  consumer(num_items,num_of_consumers);                    // call consumer

   printf("\n---------------------------------------Consumer Work ends here----------------------------------------\n");

  cout << " \n \n  Number of Items produced " << Producer::index << endl;
  cout << " \n  Number of Items consumed " << Consumer::index << endl <<endl;   // printing number of produced and consumed
                                                                                //   items  
  return 0;


 
}
