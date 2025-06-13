
/*
*********************************************************************

	C-DAC Tech Workshop : hyPACK-2013
                    October 15-18, 2013

    Example 5.8 : mm_mult_slave_self_schd.c

    Objective   : Matrix_Matrix Multiplication
                  (Self_Scheduling - Worker Program)

    Input       : A matrix row from  Master 

    Output      : Computed row result to Master 

   Description  : This is the worker program which will be executed 
                  by the rest of the processes.

                  The workers does the computations on the data 
                  distributed by the master & indicates to the master 
                  for which row it has done the computation.

   Created      : August-2013

   E-mail       : hpcfte@cdac.in     

****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

main(int argc, char** argv) 
{

 int		  Numprocs, MyRank;
 int		  Source, Destination;
 int		  Source_tag, Destination_tag;
 int		  NoofColsA, NoofRowsB, NoofColsB;
 int 		  irow, icol, index, flag;
 int		  Root = 0;
 float		  *RowA, *BufferB, *ResultRow;
 int		  MatrixA_FileStatus = 1, MatrixB_FileStatus = 1;
 MPI_Status status;     

  /* ........MPI Initialisation .......*/
 MPI_Init(&argc, &argv); 
 MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
 MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Bcast (&MatrixA_FileStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(MatrixA_FileStatus == 0) {
		 MPI_Finalize();
       exit(-1);
  }
  MPI_Bcast (&MatrixB_FileStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(MatrixB_FileStatus == 0) {
		 MPI_Finalize();
       exit(-1);
  }

	MPI_Bcast (&NoofColsA, 1, MPI_INT, 0, MPI_COMM_WORLD); 
	MPI_Bcast (&NoofRowsB, 1, MPI_INT, 0, MPI_COMM_WORLD); 
   if(NoofColsA != NoofRowsB){
		MPI_Finalize();
		exit(0);
	}
	MPI_Bcast (&NoofColsB, 1, MPI_INT, 0, MPI_COMM_WORLD); 

   BufferB = (float *)malloc(NoofRowsB*NoofColsB*sizeof(float));   
	MPI_Bcast (BufferB, NoofRowsB*NoofColsB, MPI_FLOAT, 0, MPI_COMM_WORLD); 

   ResultRow = (float *)malloc(NoofColsB*sizeof(float));   
	RowA 	    = (float *)malloc(NoofColsA*sizeof(float));   

   for(;;) {
	   MPI_Recv(RowA, NoofColsA, MPI_FLOAT, Root, MPI_ANY_TAG, 
					MPI_COMM_WORLD, &status);

      flag = status.MPI_TAG - 1;
	   if(flag == -1) 
			break;

		Destination = Root;
		Destination_tag = flag;

		for(icol = 0; icol< NoofColsB; icol++) {
			ResultRow[icol] = 0;
			for(irow=0; irow< NoofRowsB; irow++) {
				index = irow*NoofColsB + icol;
			   ResultRow[icol] += BufferB[index] * RowA[irow];
			}
		}
		MPI_Send(ResultRow, NoofColsB, MPI_FLOAT, 0, Destination_tag, MPI_COMM_WORLD);
   }
   MPI_Finalize();

   /*...Free Allocated Memory ......*/
 free(RowA);
 free(BufferB);
 free(ResultRow);
}
