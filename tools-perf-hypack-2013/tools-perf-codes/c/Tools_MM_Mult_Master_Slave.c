/*
*******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example 5   :(Tools_MM_Mult_Master_Slave.c)	

 Objective   : Matrix_Matrix_Multiplication
              (Self_Scheduling algorithm or Master Slave)

 Description : In this the processor other than rank 0 will do 
	       the computations on the data distributed by the processor 
               with rank 0 and indicates to the root for which row it has
               done the computation.

 Input       : Read Matrix 1 & Matrix 2.

 Output      : Result of Matrix-Matrix Multiplication on processor 0.

 Necessary Condition : Number of Processes should be
		       less than or equal to 8

   Created          : August-2013

   E-mail           : hpcfte@cdac.in     

*******************************************************************
*/

#include <stdio.h>
#include <math.h>
#include "mpi.h"
#define SIZE 16


main(int argc, char** argv) 
{

 int	     Numprocs, MyRank;
 int		  NoofRowsA, NoofColsA,	NoofRowsB, NoofColsB;
 int		  Source, Destination, Root=0;
 int	     Source_tag, Destination_tag;
 int 	     flag,irow, icol, iproc, index;
 int	     RowtoSend = 0, RowReceived = 0;
 float	     **Matrix_A, **Matrix_B, **ResultMatrix;
 float	     *BufferB, *RowA, *ResultRow;
 MPI_Status status;     

 /* ........MPI Initialisation .......*/
	MPI_Init(&argc, &argv); 
  	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
  	MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);

 /* Reading Input */
if(MyRank==0)
{
     NoofRowsA=SIZE;
     NoofColsA=SIZE;
     Matrix_A = (float **) malloc (NoofRowsA * sizeof(float *));
     for (irow = 0; irow < NoofRowsA; irow++){
	Matrix_A[irow] = (float *) malloc(NoofColsA * sizeof(float));
     		for (icol = 0; icol < NoofColsA; icol++)
       		Matrix_A[irow][icol]= rand(10);
     		}

     NoofRowsB=SIZE;
     NoofColsB=SIZE;
     Matrix_B = (float **) malloc (NoofRowsB * sizeof(float *));
     for(irow = 0; irow < NoofRowsB; irow++){
        Matrix_B[irow] = (float *) malloc(NoofColsB * sizeof(float *));
       		for(icol = 0; icol < NoofColsB; icol++)
     		Matrix_B[irow][icol]=rand(10);
    		}

     MPI_Barrier(MPI_COMM_WORLD);
     MPI_Bcast (&NoofColsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast (&NoofRowsB, 1, MPI_INT, 0, MPI_COMM_WORLD);

   if(NoofColsA != NoofRowsB){
    MPI_Finalize();
    printf("Incompatible dimensions of Matrices for Mutlipication\n");
    exit(0);
 }

   MPI_Bcast (&NoofColsB, 1, MPI_INT, 0, MPI_COMM_WORLD);

 /* .......Convert Matrix B into 1-D BufferB and Broadcast to Workers.....*/
 BufferB = (float *) malloc(NoofRowsB*NoofColsB*sizeof(float));
 index  = 0;
 for(irow = 0; irow<NoofRowsB; irow++)
   for(icol=0; icol<NoofColsB; icol++)
   BufferB[index++] = Matrix_B[irow][icol];
   MPI_Bcast (BufferB,NoofRowsB*NoofColsB,MPI_FLOAT,0,MPI_COMM_WORLD);

 /* .......Allocate memory for output and a row of a Matrix A......*/ 
   RowA = (float *)malloc(NoofColsA*sizeof(float));
   ResultMatrix = (float **)malloc(NoofRowsA*sizeof(float *)); 
   for(irow=0; irow<NoofRowsA; irow++)
     ResultMatrix[irow] = (float *)malloc(NoofColsB*sizeof(float));

   /* First Send At least one row to each processors */
    RowtoSend = 0;
    for(iproc = 1 ; iproc < Numprocs ; iproc++) {
	for(icol=0; icol<NoofColsA; icol++)
	RowA[icol] = Matrix_A[RowtoSend][icol];
   MPI_Send(RowA, NoofColsA, MPI_FLOAT, iproc, RowtoSend+1, MPI_COMM_WORLD);
	RowtoSend++;
	}

   ResultRow = (float *) malloc(NoofColsB*sizeof(float));
	for(irow = 0 ; irow < NoofRowsA ; irow++) {
	MPI_Recv(ResultRow, NoofColsB, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	Destination = status.MPI_SOURCE;
	Source_tag  = status.MPI_TAG;
		 
	memcpy(ResultMatrix[Source_tag], ResultRow, NoofColsB*sizeof(float));

 /*For the  remaining rows(if any) send the data to whichever processor that finishes the earlier computations first */

   if( RowtoSend < NoofRowsA ) {
	 memcpy(RowA, Matrix_A[RowtoSend], NoofColsA*sizeof(float));
 	 Destination_tag = RowtoSend+1;
   	 MPI_Send(RowA, NoofColsA, MPI_FLOAT, Destination, Destination_tag, MPI_COMM_WORLD);
	 RowtoSend++;
    }

	}

 for(iproc = 1 ; iproc < Numprocs ; iproc++) {
   MPI_Send(RowA, NoofColsA, MPI_FLOAT, iproc, 0, MPI_COMM_WORLD);
}
}
if(MyRank!=0)
{

MPI_Barrier(MPI_COMM_WORLD);

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
       RowA        = (float *)malloc(NoofColsA*sizeof(float));

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
                MPI_Send(ResultRow, NoofColsB, MPI_FLOAT, 0, Destination_tag,MPI_COMM_WORLD);
   }
}

if(MyRank==0)
{
	
  printf("Result is ...\n");
	for(irow = 0 ; irow < NoofRowsA; irow++) {
		for( icol= 0 ;  icol< NoofColsB; icol++) 
			printf(" %3.3f ",ResultMatrix[irow][icol]); 
	   printf("\n");
	
}
}
   MPI_Finalize(); 
}
