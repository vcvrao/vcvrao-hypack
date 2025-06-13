
/**********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example 1.4 : mpi-io-write-single-file.c
 
  Objective   : MPI  Parallel I/O to Single I/O 
                Based on the number of processes spawned the program
                Each process use  MPI 2.0 I/O library calls. 
                and writes to s single file                         
 
  Input       : None 

  Output      : Output files with random data written                                          
                                                                                
  

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

****************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include<stdlib.h>
#define BUFSIZE 100

int main (int argc, char *argv[])
{
	int i, myrank, numprocs, buf[BUFSIZE];
	MPI_File thefile;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size (MPI_COMM_WORLD, &numprocs);

	for (i=0; i<BUFSIZE; i++)
	     buf[i] = myrank * BUFSIZE + i;

	MPI_File_open(MPI_COMM_WORLD,  "testfile",
			  MPI_MODE_CREATE | MPI_MODE_WRONLY,
			  MPI_INFO_NULL, &thefile);

	MPI_File_set_view(thefile, myrank * BUFSIZE * sizeof(int),
			     MPI_INT, MPI_INT, "native", MPI_INFO_NULL);

	MPI_File_write(thefile, buf, BUFSIZE, MPI_INT, MPI_STATUS_IGNORE);

	MPI_File_close(&thefile);
	MPI_Finalize();
	return 0;
}



