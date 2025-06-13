
/*******************************************************************

 *	C-DAC Tech Workshop : hyPACK-2013
 *               October 15-18, 2013
 *
 *      Example 1.1 (non-mpi-par-io-multiple-files.c) 
 *
 * Objective            : write into n files using  MPI 1.X library calls
 *
 * Input                : None 
 *
 * Description          : Based on the number of processes spawned the 
 *                        program  writes to seperate 'p' files on the host 
 *                        where root process is spawned
 *
 * Output               : Output files with data written 
 *
 *                        
 *  Created            : August-2013
 *
 *  E-mail             : hpcfte@cdac.in     
 * 
 * Necessary conditions : No of Processes should be less than or equal to 8
 *
 * 
 ***********************************************************************
 */

/* Example of parallel Unix write into separate files */

#include "mpi.h" 
#include <stdio.h> 
#include<stdlib.h>
#define BUFSIZE 100

  int main(int argc, char *argv[]) 
  {

     int i, myrank,Numprocs,buf[BUFSIZE]; 
     char filename[128];
     FILE *myfile;

     MPI_Init(&argc, &argv);
     MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
     MPI_Comm_size(MPI_COMM_WORLD,&Numprocs);

     for (i = 0; i < BUFSIZE; i++)
        buf[i] = myrank * BUFSIZE + i;

     sprintf(filename, "testfile.%d.txt", myrank);

     myfile = fopen(filename, "w");
/*     fprintf(myfile,"\n Filename : %s \n My Rank %d /%d \n ",
             filename,myrank,Numprocs);
*/
     fwrite(buf, sizeof(int), BUFSIZE, myfile); 

     fclose(myfile);

     MPI_Finalize() ;
     return 0;
}
