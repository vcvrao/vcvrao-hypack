
/*******************************************************************
 *		C-DAC Tech Workshop : hyPACK-2013
 *                      October 15-18, 2013
 *
 *           Example 1.6 (Reading the file with a different processes) 
 *
 * Objective            : MPI  Parallel I/O read 
 *
 * Input                : None 
 *
 * Description          : MPI  Parallel I/O to Multiple files
 *                        Based on the number of processes spawned the program
 *                        Each process use  MPI 2.0 I/O library calls. 
 *                        and reads differnt file                     
 *
 * Output               : None (Reading a file)  
 *
 * Necessary conditions : Number of Processes should be less than or equal to 8
 * 
 * Created             : August-2013
 *
 * E-mail              : hpcfte@cdac.in     

 ***********************************************************************
 */

/* Parallel MPI read with arbitrary number of processes */

#include "mpi.h"
#include <stdio.h>
#include<stdlib.h>
int main (int argc, char *argv[])
{
	int i, myrank, numprocs, bufsize, *buf, count;
	MPI_File thefile;
	MPI_Status status;
	MPI_Offset filesize;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_File_open(MPI_COMM_WORLD, "testfile.txt", MPI_MODE_RDONLY,
			  MPI_INFO_NULL, &thefile);

	MPI_File_get_size(thefile, &filesize);     /* in bytes */
	filesize = filesize / sizeof(int);         /* in number of ints */
	bufsize  = filesize / numprocs + 1;	   /* local number to read */

	buf = (int *) malloc(bufsize * sizeof(int));

	MPI_File_set_view(thefile, myrank * bufsize * sizeof(int),
			    MPI_INT, MPI_INT, "native", MPI_INFO_NULL);

	MPI_File_read(thefile, buf, bufsize, MPI_INT, &status);

	MPI_Get_count(&status, MPI_INT, &count);

	printf("process %d read %d ints\n", myrank, count);

	MPI_File_close(&thefile);

	MPI_Finalize();
	return 0;
}


