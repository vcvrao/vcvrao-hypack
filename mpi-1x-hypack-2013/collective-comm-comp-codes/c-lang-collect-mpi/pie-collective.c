
/*
**********************************************************************
		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example 2.8	    : pie_collective.c

  Objective           : To compute the value of PI by numerical integration.
                        MPI Collective communication library calls are used.

                        This example demonstrates the use of 
                        MPI_Init
                        MPI_Comm_rank
                        MPI_Comm_size
                        MPI_Bcast
                        MPI_Reduce
                        MPI_Finalize

  Input               : The number of intervals.

  Output              : The calculated value of PI.

  Necessary Condition : Number of Processes should be
                        less than or equal to 8.

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

**********************************************************************
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

double func(double x)
{
    return (4.0 / (1.0 + x*x));
}

int main(int argc,char *argv[])
{
    int    NoInterval, interval;
    int    MyRank, Numprocs, Root ; 
    double mypi, pi, h, sum, x;
    double PI25DT = 3.141592653589793238462643;

    /*....MPI initialisation....*/
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&Numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);

    if(MyRank == Root){
	puts("\nEnter the number of intervals : ");
	scanf("%d",&NoInterval);
    }
        
   /*....Broadcast the number of subintervals to each processor....*/ 
   MPI_Bcast(&NoInterval, 1, MPI_INT, 0, MPI_COMM_WORLD);

   if(NoInterval <= 0){
      if(MyRank == Root) 
	 printf("Invalid Value for Number of Intervals .....\n");
      MPI_Finalize();
      exit(-1);
   }

   h   = 1.0 / (double)NoInterval;
   sum = 0.0;
   for(interval = MyRank + 1; interval <= NoInterval; interval += Numprocs){
       x = h * ((double)interval - 0.5);
       sum += func(x);
   }
   mypi = h * sum;

   /*....Collect the areas calculated in P0....*/ 
   MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, Root, MPI_COMM_WORLD);

   if(MyRank == Root){
      printf("pi is approximately %.16f, Error is %.16f\n", 
						   pi, fabs(pi - PI25DT));
   }

   MPI_Finalize();

}




