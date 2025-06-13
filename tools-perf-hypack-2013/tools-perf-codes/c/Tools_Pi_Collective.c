/*
**********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example 3    :(Tools_Pi_Collective.c)	

  Objective   : To compute the value of PI by numerical integration.
                MPI Collective communication library calls are used.

		This example demonstrates the use of MPI_Init

                MPI_Comm_rank
                MPI_Comm_size
                MPI_Bcast
                MPI_Reduce
                MPI_Finalize

  Input       : The number of intervals.

  Output      : The calculated value of PI.

  Necessary Condition : Number of Processes should be
                        less than or equal to 8.

  Created             : August-2013

  E-mail              : hpcfte@cdac.in     


**********************************************************************
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
    double result, pi, h, sum, x;
    int  namelen;

    /* MPI Initialization */

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&Numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);
   /*  Initializing No of Intervals */ 
       NoofIntervals = 0;
        if (MyRank == 0)
        {
	    if (NoofIntervals==0) NoofIntervals=10000; else NoofIntervals=0;
        }
        MPI_Bcast(&NoofIntervals, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (NoofIntervals == 0)
        MPI_Finalize();
        else
        {
            h   = 1.0 / (double) NoofIntervals;
            sum = 0.0;
	    /* A slightly better approach starts from large i and works back */
            for (i = MyRank + 1; i <= NoofIntervals; i += Numprocs)
            {
                x = h * ((double)i - 0.5);
                sum += f(x);
            }
            result = h * sum;
         /* Summing up of all the results to a pi */
            MPI_Reduce(&result, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (MyRank == 0)
	    {
                printf("pi is approximately %.16f, Error is %.16f\n",
                       pi, fabs(pi - PI25DT));
	    }
        }
    MPI_Finalize();
    return 0;
}
