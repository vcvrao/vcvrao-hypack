/*
*******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example 5.8	      : mm_mult_master_self_schd.c

  Objective           : Matrix_Matrix_Multiplication
                        (Self_Scheduling-Master Program)

  Input               : Read files (mdata1.inp) for Matrix 1
                        & (mdata2.inp) for Matrix 2.

  Output              : Result of Matrix-Matrix Multiplication on process 0.

  Necessary Condition : Number of Processes should be
                        less than or equal to 8

   Created            : August-2013

   E-mail             : hpcfte@cdac.in     

*****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

main(int argc, char** argv) 
{

 int	     Numprocs, MyRank;
 int		  NoofRowsA, NoofColsA,	NoofRowsB, NoofColsB;
 int		  Source, Destination, Root=0;
 int	     Source_tag, Destination_tag;
 int 	     irow, icol, iproc, index;
 int	     RowtoSend = 0, RowReceived = 0;
 float	     **Matrix_A, **Matrix_B, **ResultMatrix;
 float	     *BufferB, *RowA, *ResultRow;
 FILE	     *fp;
 int		  MatrixA_FileStatus = 1, MatrixB_FileStatus = 1;
 MPI_Status status;     

 /* ........MPI Initialisation .......*/
	MPI_Init(&argc, &argv); 
  	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
  	MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);

 /* Reading Input */

    if ((fp = fopen ("./data/mdata1.inp", "r")) == NULL){
			MatrixA_FileStatus = 0;
     }

  if(MatrixA_FileStatus != 0) {
     fscanf(fp, "%d %d\n", &NoofRowsA, &NoofColsA);

     Matrix_A = (float **) malloc (NoofRowsA * sizeof(float *));
     for (irow = 0; irow < NoofRowsA; irow++){
	Matrix_A[irow] = (float *) malloc(NoofColsA * sizeof(float));
     		for (icol = 0; icol < NoofColsA; icol++)
       		fscanf(fp, "%f", &Matrix_A[irow][icol]);
     		}
     		fclose(fp);
	 }

     if((fp = fopen ("./data/mdata2.inp", "r")) == NULL){
			MatrixB_FileStatus = 0;
     }

  if(MatrixB_FileStatus != 0) {
	fscanf(fp, "%d %d\n", &NoofRowsB, &NoofColsB);

     	Matrix_B = (float **) malloc (NoofRowsB * sizeof(float *));
     for(irow = 0; irow < NoofRowsB; irow++){
      Matrix_B[irow] = (float *) malloc(NoofColsB * sizeof(float *));
       		for(icol = 0; icol < NoofColsB; icol++)
     		fscanf(fp, "%f", &Matrix_B[irow][icol]);
    		}
     		fclose(fp);
	  }

   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Bcast (&MatrixA_FileStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if(MatrixA_FileStatus == 0) {
       printf("Can't open input file for Matrix A ......");
		 MPI_Finalize();
       exit(-1);
   }
   MPI_Bcast (&MatrixB_FileStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if(MatrixB_FileStatus == 0) {
       printf("Can't open input file for Matrix B ......");
		 MPI_Finalize();
       exit(-1);
   }

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
     ResultMatrix[irow] = (float *) malloc(NoofColsB*sizeof(float));

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

 for(iproc = 1 ; iproc < Numprocs ; iproc++) 
   MPI_Send(RowA, NoofColsA, MPI_FLOAT, iproc, 0, MPI_COMM_WORLD);
	
  printf("Result is ...\n");
	for(irow = 0 ; irow < NoofRowsA; irow++) {
		for( icol= 0 ;  icol< NoofColsB; icol++) 
			printf("  %3.3f   ",ResultMatrix[irow][icol]); 
	   printf("\n");
	}
		
	/*...Free Allocated Memory......*/
   for(irow=0; irow<NoofRowsA; irow++)
		free(Matrix_A[irow]);
   for(irow=0; irow<NoofRowsB; irow++)
		free(Matrix_B[irow]);
   for(irow=0; irow<NoofRowsA; irow++)
		free(ResultMatrix[irow]);

   free(Matrix_A);
   free(Matrix_B);
   free(ResultMatrix);
   free(RowA);
   free(BufferB);
   free(ResultRow);
   fclose(fp);

   MPI_Finalize(); 
}




