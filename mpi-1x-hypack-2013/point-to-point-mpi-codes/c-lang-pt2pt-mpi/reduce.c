/*
********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

   Example 10	       : reduce.c

  Objective            : To find sum of 'n' integers on 'p' processors
                         using MPI collective communication and communciation
                         library call ( MPI_REDUCE ).

                         This example demonstrates the use of 
                         MPI_Init
                         MPI_Comm_rank
                         MPI_Comm_size
                         MPI_Reduce
                         MPI_Finalize

  Input               : Automatic generation of input
                        The rank of each proceess is input on each process.

  Output              : Process with Rank 0 should print the sum of 'n' values.

  Necessary           : Number of Processes should be
  Condition             less than or equal to 8.

  Created             : August-2013

  E-mail              : hpcfte@cdac.in     


**********************************************************************
*/


#include <stdio.h>
#include "mpi.h"

int main(int argc,char *argv[])
{
    int	   sum = 0;
    int    MyRank, Numprocs, Root = 0;
    MPI_Status status;

    /*....MPI Initialisation....*/

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&Numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);

    /*....The REDUCE function of MPI....*/

    MPI_Reduce(&MyRank, &sum, 1, MPI_INT, MPI_SUM, Root, MPI_COMM_WORLD);

    if(MyRank == Root) 
		 printf("SUM = %d\n", sum);

    MPI_Finalize();
}




