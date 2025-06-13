
/*****************************************************************

	C-DAC Tech Workshop : hyPACK-2013
                  October 15-18, 2013

    Example 2.2		: scatter.c 

    Objective           : To Scatter an integer array of size "n by 1"        
                          using MPI Collective communication library call 
                          (MPI_Scatter)

                          This example demonstrates the use of 
                          MPI_Init
                          MPI_Comm_rank
                          MPI_Comm_size
                          MPI_Bcast
                          MPI_Scatter
                          MPI_Finalize

    Input               : Input Data file "sdata.inp".

    Output              : Print the scattered array on all processes.

    Necessary Condition : Number of processes should be
                          less than or equal to 8.

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi.h"

main(int argc, char** argv) 
{

  /* .......Variables Initialisation ......*/
  int        Numprocs, MyRank, Root = 0;
  int        index;
  int        *InputBuffer, *RecvBuffer;
  int        Scatter_DataSize; 
  int        DataSize;
  FILE       *fp;
  MPI_Status status;     

  /* ........MPI Initialisation .......*/
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
  MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);

  if(MyRank == Root) {

    /* .......Read the Input file ......*/

    if ((fp = fopen ("./data/sdata.inp", "r")) == NULL)
    {
       printf("\nCan't open input file");
       exit(-1);
    }

    fscanf(fp, "%d\n", &DataSize);     

    /* ...Allocate memory and read data .....*/

    InputBuffer = (int *)malloc(DataSize * sizeof(int));

    for(index=0; index< DataSize; index++) 
        fscanf(fp, "%d", &InputBuffer[index]);     
  }	

  MPI_Bcast(&DataSize, 1, MPI_INT, Root, MPI_COMM_WORLD);
  if(DataSize % Numprocs != 0) { 
     if(MyRank == Root)
        printf("Input is not evenly divisible by Number of Processes\n");
     MPI_Finalize();
     exit(-1);
  }

  Scatter_DataSize = DataSize / Numprocs;
  RecvBuffer= (int *)malloc(Scatter_DataSize * sizeof(int));

  MPI_Scatter(InputBuffer, Scatter_DataSize, MPI_INT, RecvBuffer, 
	      Scatter_DataSize, MPI_INT, Root, MPI_COMM_WORLD); 

  for(index = 0; index < Scatter_DataSize; ++index)
      printf("MyRank = %d, RecvBuffer[%d] = %d \n", 
	      MyRank, index, RecvBuffer[index]);

 MPI_Finalize();

}




