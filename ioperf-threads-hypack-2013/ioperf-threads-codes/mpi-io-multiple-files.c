
/*******************************************************************
 *		C-DAC Tech Workshop : hyPACK-2013
 *                           October 15-18, 2013
 *
 *                       Example 1.3 (mpi-io-multiple-files.c) 
 *
 * Objective            : Open, write into n files using MPI 2.X parallel
 *                        I/O lib calls 
 *
 * Input                : None 
 *
 * Description          : Based on the number of processes spawned the 
 *                        program  writes to seperate 'p' files on the host 
 *                        where root process is spawned
 *
 * Output               : Output files with data written 
 * 
 * Created             : August-2013
 *
 * E-mail              : hpcfte@cdac.in     
 *
 * Necessary conditions : Number of Processes should be less than or equal to 8
 ***********************************************************************
 */

#include "mpi.h"
#include <stdio.h>
#define BUFSIZE 10
#define length 15
#include<string.h>
int
main(int argc, char **argv)
{

	int             i, MyRank,numprocs;
	char  		buf[BUFSIZE],file_name[122];
	char            *filename;
	MPI_File        myfile;

        /* .... MPI Initialization .... */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    	MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);
	filename = (char *) malloc(length * sizeof(char));
	strcpy(filename, "Parallelio");
	for (i = 0; i < BUFSIZE; i++)
	{
               	buf[i] = 'A';
 	}
	printf("\n rank=%d",MyRank);
	sprintf(file_name,"%d",MyRank);
	strcat(filename,file_name);
	printf("rank : %d, filename : %s",MyRank,filename);
	if(MyRank<numprocs)
	{
		MPI_File_open(MPI_COMM_SELF,filename , MPI_MODE_RDWR | MPI_MODE_CREATE,MPI_INFO_NULL, &myfile);
		MPI_File_write(myfile, buf, BUFSIZE, MPI_CHAR, MPI_STATUS_IGNORE);
		MPI_File_close(&myfile);
	}
        /* ..... MPI Finalization .... */
	MPI_Finalize();
	return 0;
}
