/*************************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

        Example    : main.cc

        Objective  : To explain the producer consumer work queue problem using Intel TBB 
     
        Input      : Number of Items, Number of Producers, Number of Consumers

        Output     : Sequence of Producer Consumer work and time taken 
                    to compute the matrix multiplication

       Created     : August-2013

      E-mail       : hpcfte@cdac.in     
******************************************************************************/



#include"../include/headerfiles.h"                               // including header files
#include"../include/Proto.h"

int main (int argc,char **argv) 
{
  
  if (argc != 4)
   {
      printf ("Usage: executable # num_items #num_of_producers #num_of_consumers \n");
      return 1;
   }


  int num_items =  atoi (argv[1]);
  int num_of_producers = atoi (argv[2]);
  int num_of_consumers = atoi (argv[3]);


  tbb::task_scheduler_init init;



   printf("\n---------------------------------------Producer Work starts here----------------------------------------\n");
   
   tick_count t0 = tick_count::now();
 
   producer(num_items,num_of_producers);                      // call producer

   printf("\n---------------------------------------Producer Work ends here----------------------------------------\n");

   printf("\n---------------------------------------Consumer Work starts here----------------------------------------\n");

   consumer(num_items,num_of_consumers);                    // call consumer

   tick_count t1 = tick_count::now();

   printf("\n---------------------------------------Consumer Work ends here----------------------------------------\n");

    double t_exetime = (t1-t0).seconds();

    printf("\n\n Time required for execution  =  %g\n\n",t_exetime);

   return 0;

 
}
