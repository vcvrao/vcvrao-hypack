
/*
******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

    Example New		:  gather.c

    Objective           : To gather  an integer array of size "n: from 
                          each process 
                          using MPI Collective communication library call 
                          (MPI_Reduce and MPI Gather )

                          This example demonstrates the use of 
                          MPI_Init
                          MPI_Comm_rank
                          MPI_Comm_size
                          MPI_Reduce
                          MPI_Gather 
                          MPI_Finalize

    Input               : Input Data file "mygather0.inp", "mygather1.inp".

    Output              : Print the gather array on process with rank 0 .

    Necessary          : Number of processes should be equal to 2.

   Created             : September 2019 

   E-mail              : vcvraocdac.in     


*******************************************************************
*/



#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

main(int argc, char** argv) 
{

  /* .......Variables Initialisation ......*/
  int        Numprocs, MyRank, Root = 0;
  int        index;
  int        *InputBuffer, *RecvBuffer;
  int        Scatter_DataSize; 
  int        DataSize;
  FILE       *fp0;
  FILE       *fp1;
  MPI_Status status;     

  /* ........MPI Initialisation .......*/
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
  MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);

  if(MyRank == 0) {

    /* .......Read the Input file from proceee 0 ......*/

    if ((fp0 = fopen ("./data/mygather0.inp", "r")) == NULL)
    {
       printf("\nCan't open input file");
       exit(-1);
    }

    fscanf(fp0, "%d\n", &DataSize);     

    /* ...Allocate memory and read data .....*/

    InputBuffer = (int *)malloc(DataSize * sizeof(int));

    for(index=0; index< DataSize; index++) 
        fscanf(fp0, "%d", &InputBuffer[index]);     

/*   printf("MyRank = %d, DataSize  = %d \n", MyRank, DataSize); */

    fclose(fp0);
  }	


  if(MyRank == 1) {

    /* .......Read the Input file from proceee 0 ......*/

    if ((fp1 = fopen ("./data/mygather1.inp", "r")) == NULL)
    {
       printf("\nCan't open input file");
       exit(-1);
    }

    fscanf(fp1, "%d\n", &DataSize);     

    /* ...Allocate memory and read data .....*/

    InputBuffer = (int *)malloc(DataSize * sizeof(int));

    for(index=0; index< DataSize; index++) 
        fscanf(fp1, "%d", &InputBuffer[index]);     

    fclose(fp1);
  }	

   printf("MyRank = %d, DataSize  = %d \n", MyRank, DataSize);

 MPI_Finalize();

}




