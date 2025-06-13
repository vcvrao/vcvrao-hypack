/*
*********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example 4 : (Tools_Pi_Idleness.c)	

  Objective : To compute the value of PI by numerical integration.
              MPI Point-to-Point communication and collective communication 
              library calls are used to show the idleness in a program.

	      This example demonstrates the use of 
	      MPI_Init
              MPI_Comm_rank
              MPI_Comm_size
              MPI_Bcast
              MPI_Send
              MPI_Recv
              MPI_Finalize

  Input     : The number of intervals.

  Output    : The calculated value of PI.

  Necessary Condition : Number of Processes should be less 
                        than or equal to 8.

 Created             : August-2013

 E-mail              : hpcfte@cdac.in     


*********************************************************************
*/

#include "mpi.h"
#include <stdio.h>
#include <math.h>


double f(double);

double f(double a)
{
    return (4.0 / (1.0 + a*a));
}

int main(int argc,char *argv[])
{
    int done = 0, NoofIntervals, MyRank, Numprocs, i;
    double PI25DT = 3.141592653589793238462643;
    double mypi=0, pi, h, sum, x;
    double startwtime = 0.0, endwtime;
    int  namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;
    
    /* Initializing MPI CALLS */
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&Numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);


    NoofIntervals = 0;
        if (MyRank == 0)
        {
	    if (NoofIntervals==0) NoofIntervals=10000; else NoofIntervals=0;

        }
       /* broadcasting no of intervals to all the processors */
        MPI_Bcast(&NoofIntervals, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (NoofIntervals == 0)
            done = 1;
        else
        {
         if(MyRank==0)
           {
            h   = 1.0 / (double) NoofIntervals;
            sum = 0.0;
             for (i = MyRank + 1; i <= NoofIntervals; i += Numprocs)
             {
              x = h * ((double)i - 0.5);
             sum += f(x);
             }
              mypi= h * sum;  
            MPI_Send(&mypi,1,MPI_DOUBLE,MyRank+1,0,MPI_COMM_WORLD);
          }
          else
           { 
             if(MyRank< Numprocs-1)
             { 
            MPI_Recv(&mypi,1,MPI_DOUBLE,MyRank-1,0,MPI_COMM_WORLD,&status);
            h   = 1.0 / (double) NoofIntervals;
            sum = 0.0;
            for (i = MyRank + 1; i <= NoofIntervals; i += Numprocs)

            {
                x = h * ((double)i - 0.5);
                sum += f(x);
            }
            mypi+= h * sum;
            MPI_Send(&mypi,1,MPI_DOUBLE,MyRank+1,0,MPI_COMM_WORLD); 
            }

            if (MyRank == Numprocs-1)
	    {
              MPI_Recv(&mypi,1,MPI_DOUBLE,MyRank-1,0,MPI_COMM_WORLD,&status);
                h   = 1.0 / (double) NoofIntervals;
                sum = 0.0;
                for (i = MyRank + 1; i <= NoofIntervals; i += Numprocs)
                {
                   x = h * ((double)i - 0.5);
                   sum += f(x);
                }
                mypi+= h * sum;
           /* displaying the final result */
                printf("pi is approximately %.16f, Error is %.16f\n",
                       mypi, fabs(mypi - PI25DT));
		fflush( stdout );
	    }
           } 
        
    }
    MPI_Finalize();
}
