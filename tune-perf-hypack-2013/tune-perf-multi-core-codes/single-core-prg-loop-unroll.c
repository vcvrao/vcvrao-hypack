/*****************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example 1.1           : single-core-prg-loop-unroll.c

 Objective 	       : Write a program to demonstrate the execution time for the 
                         following loop with/without Loop Unrolling.
                         for (i=0; i<n; i++)
                          a[i]= a[i] <b[i] ? b[i] :c[i];

 Input                 : Size of the vectors.

 Output                : Time taken in microseconds by the loop with/without
                         unrolling.

  Created             : August-2013

  E-mail              : hpcfte@cdac.in     

*********************************************************************************/


#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#define seed 10000

   main(int argc, char *argv[]) {

   int CounterI, Size, Remain;
   int VectorAorig[10000], VectorBorig[10000], VectorCorig[10000];
   int VectorAmod[10000], VectorBmod[10000], VectorCmod[10000];

   struct timeval  TimeValue_Start;
   struct timezone TimeZone_Start;
   struct timeval  TimeValue_Final;
   struct timezone TimeZone_Final;
   double time_start, time_end, time_overheadorig,time_overheadmod;

     
   if (argc<2){
        printf(" Please provide a positive integer for the size of the vectors as command-line argument\n");
        return (-1);
        }
   
   Size=atoi(argv[1]);
   Size=abs(Size);
   if (Size>10000){
        printf(" Please provide a positive integer less than equal to 10000 for the size of the vectors\n");
        return (-1);
        }
   printf("\n Size is taken as %d", Size);

/* Populating the matrices B and C, Matrix A consists of random values */
  
   srand(seed);
   for(CounterI=0; CounterI<Size; CounterI++) {
        VectorAorig[CounterI]=VectorAmod[CounterI]=rand();
        VectorBorig[CounterI]=VectorBmod[CounterI]=100;
        VectorCorig[CounterI]=VectorCmod[CounterI]=200;
        }


/* Calculating time for the loop with Loop UnRolling */

   Remain=Size%4;
   gettimeofday(&TimeValue_Start, &TimeZone_Start);

/* Pre-conditioning loop */

   for(CounterI=0;CounterI<Remain;CounterI++) {
        VectorAmod[CounterI]=VectorAmod[CounterI]<VectorBmod[CounterI]?
            VectorBmod[CounterI]:VectorCmod[CounterI];
        }
   
   
   for(CounterI=Remain;CounterI<Size;CounterI=CounterI+4) {
        VectorAmod[CounterI]=VectorAmod[CounterI]<VectorBmod[CounterI]?
            VectorBmod[CounterI]:VectorCmod[CounterI];
        VectorAmod[CounterI+1]=VectorAmod[CounterI+1]<VectorBmod[CounterI+1]?
            VectorBmod[CounterI+1]:VectorCmod[CounterI+1];
        VectorAmod[CounterI+2]=VectorAmod[CounterI+2]<VectorBmod[CounterI+2]?
            VectorBmod[CounterI+2]:VectorCmod[CounterI+2];
        VectorAmod[CounterI+3]=VectorAmod[CounterI+3]<VectorBmod[CounterI+3]?
            VectorBmod[CounterI+3]:VectorCmod[CounterI+3];
        }
   gettimeofday(&TimeValue_Final, &TimeZone_Final);
   time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
   time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
   time_overheadmod = (time_end - time_start) ;

/* Calculating time for the loop without Loop UnRolling */

   gettimeofday(&TimeValue_Start, &TimeZone_Start);
   for(CounterI=0; CounterI<Size; CounterI++) {
        VectorAorig[CounterI]=VectorAorig[CounterI]<VectorBorig[CounterI]?
            VectorBorig[CounterI]:VectorCorig[CounterI];
        }
   gettimeofday(&TimeValue_Final, &TimeZone_Final);
   time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
   time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
   time_overheadorig = (time_end - time_start) ;


   for(CounterI=0; CounterI<Size; CounterI++){
        if(VectorAorig[CounterI]!=VectorAmod[CounterI]) {
            printf("The operations done by the loop with and without Loop Unrolling are not same\n");
            exit(-1);
        }
   }
   
   printf("\n\n The time(us) taken without Loop Unrolling is :%lf\n",time_overheadorig);
   printf("\n The time(us) taken with Loop Unrolling is    :%lf\n\n",time_overheadmod);
   
   if(time_overheadorig <= time_overheadmod)
   printf("\n\n Try for larger sizes to see the effect of Loop UnRolling\n\n");
   
   }
   
   

