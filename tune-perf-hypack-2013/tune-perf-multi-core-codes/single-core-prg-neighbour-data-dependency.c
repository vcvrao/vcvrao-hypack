/*****************************************************************************

	     C-DAC Tech Workshop : hyPACK-2013
                      October 15-18, 2013

                       
 Example 1.5           : single-core-prg-neighbour-data-dependency.c


 Objective 	       : Write a program to demonstrate the execution time for the 
                         following fragment of code with/without removing  neighbour
                         data dependency.
                         jwrap = array_size-1;
                         for(i=0; i <array_size; i++)
                         {
                           b[i] = (a[i] + a[jwrap] *0.5;
                           jwrap = i;
                         }
 
 Input                 : Size of the vectors.

 Output                :Time taken in microseconds by the loop with/without
                        Neighbour Data Dependency.


   Created             : August-2013

   E-mail              : hpcfte@cdac.in     


*********************************************************************************/
 
#include <stdio.h>
#include <sys/time.h>

   main() {

   int array_size,CounterI,jwrap;
   float VectorAorig[10000],VectorBorig[10000];
   float VectorAmod[10000],VectorBmod[10000];
   struct timeval  TimeValue_Start;
   struct timezone TimeZone_Start;
   struct timeval  TimeValue_Final;
   struct timezone TimeZone_Final;
   double time_start, time_end, time_overheadorig,time_overheadmod;
  
   printf("Enter the size of the vectors\n");
   scanf("%d",&array_size);
   if(array_size<1) {
        printf("The size of the Vectors should be greater than zero\n");
        exit(-1);
   }


/* Populating the vectors A and B, variables with orig suffix are used 
   in original construct and that with mod are used in modified construct */  
   for(CounterI=0;CounterI<array_size;CounterI++){
        VectorAorig[CounterI]=VectorAmod[CounterI]=100.0;
        VectorBorig[CounterI]=VectorBmod[CounterI]=200.0;
   }
   
/* Calculate time taken in microseconds for the loop with Neighbour Data 
   Dependency */

   gettimeofday(&TimeValue_Start, &TimeZone_Start);
   jwrap = array_size-1;
   for(CounterI=0; CounterI<array_size; CounterI++) {
        VectorBorig[CounterI] = (VectorAorig[CounterI]+VectorAorig[jwrap])*0.5;
        jwrap = CounterI;
   }
   gettimeofday(&TimeValue_Final, &TimeZone_Final);
   time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
   time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
   time_overheadorig = (time_end - time_start) ;

/* Calculate time taken in microseconds for the loop without Neighbour Data
   Dependency */

   gettimeofday(&TimeValue_Start, &TimeZone_Start);
   VectorBmod[0] = (VectorAmod[0] + VectorAmod[array_size-1])*0.5;
   for(CounterI=1; CounterI<array_size; CounterI++)
   VectorBmod[CounterI]=(VectorAmod[CounterI] + VectorAmod[CounterI-1])*0.5;
   gettimeofday(&TimeValue_Final, &TimeZone_Final);
   time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
   time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
   time_overheadmod = (time_end - time_start) ;

   for(CounterI=0; CounterI<array_size; CounterI++){
        if(VectorBorig[CounterI]!=VectorBmod[CounterI]) {
            printf("The operations done by the loop with and without Neighbour Data Dependency are not same\n");
            exit(-1);
        }
   }

  
   printf("\n\nThe time taken (in us) without Induction Variable Elimination is :%lf\n", time_overheadorig);
   printf("The time taken (in us) with Induction Variable Elimination is    :%lf\n\n", time_overheadmod);
 
   if(time_overheadorig <= time_overheadmod)
        printf("\n\nTry for larger sizes to see the effect of Induction Variable Elimination\n\n");
  
     
   
   }
   
   
