
/******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

    Example 2.7		: Allreduce.c

    Objective           : To broadcast an integer array of size "n" by 
                          process with rank "0"  and perform Global 
                          summation using MPI_Allreduce (Combine values 
                          from all processes and distributed the result
                          back to all processes)
                        
                          This example demonstrates the use of

                          MPI_Init
                          MPI_Comm_rank
                          MPI_Comm_size
                          MPI_Bast
                          MPI_Allreduce                      
                          MPI_Finalize

    Input               : Input Data file "sdata.inp" by proces with Rank 0

    Output              : Print the final sum on all processes.

    Necessary Condition : Number of processes should be 
                          less than or equal to 8.

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

*******************************************************************/

#include<stdio.h>
#include "mpi.h"
main(int argc ,char **argv)
{
	int rank, size;
 	int no,sum;	


	MPI_Status status;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	
	no=rank;		
	MPI_Allreduce(&no,&sum,2,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

	
	printf("Final sum is=  %d\n",sum);
	
	MPI_Finalize();
}

			



		
	

	








		

