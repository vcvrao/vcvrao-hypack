
/*
 ************************************************************************
 *
 *		C-DAC Tech Workshop : hyPACK-2013
 *                     October 15-18, 2013
 *
 *                   Example 1.1 (remote-memory-access.c) 
 *
 * Objective            : Calculate sum of n integers using Remote Memory 
 *                        Access 
 *
 * Input                : None  
 *
 * Description          : Based on the number of processes spawned the program
 *                        caclulates the sum of the first n integers where 
 *                        n=number of processes spawned
 *
 * Output               : Sum of first n integers where n=Numprocs 
 *
 * Necessary conditions : No of Processes should be less than or equal to 8 
 *
 *  Created             : August-2013
 *
 *  E-mail              : hpcfte@cdac.in     
 * 
 ************************************************************************
 */

#include<stdio.h>
#include <stdlib.h>
#include "mpi.h"

int 
main(int argc, char **argv)
{
	int             MyRank, i, Numprocs, Root = 0;
	MPI_Win         win;
	MPI_Info        info = MPI_INFO_NULL;
	int             assert;
	int             myvar, sum;
	int            *sum_arr;

        /* ..... MPI Initialization ......  */

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
	MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);

	if(Numprocs > 8)
	{
		if(MyRank == Root)
			printf("Number of processors should be less than or equal to 8 \n");
		MPI_Finalize();
		exit(-1);
	}

        /* ..... Window Creation ........... */
	if (MyRank == Root) {
		sum_arr = (int *) malloc(Numprocs * sizeof(int));
		MPI_Win_create(sum_arr, Numprocs * sizeof(MPI_INT), sizeof(MPI_INT), info, MPI_COMM_WORLD, &win);
	} else {
		MPI_Win_create(sum_arr, 0, sizeof(MPI_INT), info, MPI_COMM_WORLD, &win);
	}

	assert = 0;
	MPI_Win_fence(assert, win);
	myvar = MyRank + 1;

        /* Accumulating n integers at root process */
	MPI_Put(&myvar, 1, MPI_INT, Root, MyRank, 1, MPI_INT, win);
MPI_Win_fence(assert, win);

	if (MyRank == Root) {
		printf("The numbers accumulated at root process are:\n");
		for (i = 0; i < Numprocs; i++)
			printf("%d ", sum_arr[i]);
		printf("\n");
		sum = 0;
		for (i = 0; i < Numprocs; i++)
			sum += sum_arr[i];
		printf("Sum of first %d integers is : %d\n", Numprocs, sum);
		free(sum_arr);
	}
	MPI_Win_free(&win);

        /* .......  MPI Finalization .......... */

	MPI_Finalize();
	
	return(0);

}
