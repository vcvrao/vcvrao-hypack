/*
*******************************************************************
 *
 *		C-DAC Tech Workshop : hyPACK-2013
 *                           October 15-18, 2013
 *
 *  Example 1.2         : (mpi-io-multiple-files.c) 
 *
 * Objective            : Open, write into n files using Parallel I/O 
 *
 * Input                : None 
 *
 * Description          : Based on the number of processes spawned the 
 *                        program  writes to seperate 'p' files on the host 
 *                        where root process is spawned
 *
 * Output               : Output files with data written 
 *
 * Necessary conditions : Number of Processes should be less than or equal to 8
 *
 *   E-mail             : hpcfte@cdac.in     
 *
 *  Created             : August 2013 
 *
 ***********************************************************************
 */

#include "mpi.h"
#include <stdio.h>
#define BUFSIZE 100

int
main(int argc, char **argv)
{

	int             i, MyRank, buf[BUFSIZE];
	char            filename[128];
	MPI_File        myfile;

        /* .... MPI Initialization .... */

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

	for (i = 0; i < BUFSIZE; i++)
		buf[i] = MyRank *BUFSIZE + i;

	sprintf(filename, "ParallelIO.%d", MyRank);

        /* Opening and Writing Numprocs files using Parallel I/O */
	MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE,
		      MPI_INFO_NULL, &myfile);

	MPI_File_write(myfile, buf, BUFSIZE, MPI_INT, MPI_STATUS_IGNORE);

	MPI_File_close(&myfile);

        /* ..... MPI Finalization .... */
	MPI_Finalize();
	return 0;
}
