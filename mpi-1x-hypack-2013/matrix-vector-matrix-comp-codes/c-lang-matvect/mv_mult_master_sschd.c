/*
********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                       October 15-18, 2013

   Example 5.5         : mv_mult_master_sschd.c

   Objective           : Matrix_Vector Multiplication
                        (Self_scheduling algorithm master program)

   Input               : Process 0 (master) reads files (mdata.inp) 
                         for Matrix and (vdata.inp) for Vector

   Output              : Process 0 prints the result of Matrix_Vector 
                         Multiplication 

   Necessary Condition : Number of processors should be greater than 
                         2 and less than or equal to 8

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


main(int argc, char** argv) 
{

   
  int	    Numprocs, MyRank;
  int 	    NoofCols, NoofRows, VectorSize; 
  int	    index, irow, icol, iproc, ValidInput=1;
  int	    Root=0, Source, Destination;
  int	    Source_tag, Destination_tag;
  int	    RowtoSend = 0;
  float     **Matrix, *Buffer, Sum;
  float     *Vector, *FinalVector;
  float     *CheckResultVector;
  FILE	    *fp;
  int	    MatrixFileStatus = 1, VectorFileStatus = 1;
  MPI_Status status;     


  /* ........MPI Initialisation .......*/

  MPI_Init(&argc, &argv); 
  MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
  MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);

  if(Numprocs < 2) {
     printf("Invalid Number of Processors ..... \n");
     printf("Numprocs must be greater than 1 ......\n");
     MPI_Finalize();
     exit(0);
  }  	

  /* .......Read the Input file ......*/
  if ((fp = fopen ("./data/mdata.inp", "r")) == NULL){
       MatrixFileStatus = 0;
  }

  if(MatrixFileStatus != 0) {
     fscanf(fp, "%d %d\n", &NoofRows,&NoofCols);

    /* ...Allocate memory and read Matrix from file .......*/
     Matrix = (float **)malloc(NoofRows*sizeof(float *));
     for(irow=0 ;irow<NoofRows; irow++){
 	 Matrix[irow] = (float *)malloc(NoofCols*sizeof(float));
 	 for(icol=0; icol<NoofCols; icol++) {
    	    fscanf(fp, "%f", &Matrix[irow][icol]);     
       	 }
    }
    fclose(fp);

  }

  /* Read vector from input file */
  if ((fp = fopen ("./data/vdata.inp", "r")) == NULL){
       VectorFileStatus = 0;
  }

  if(VectorFileStatus != 0) {
     fscanf(fp, "%d\n", &VectorSize);

     Vector = (float*)malloc(VectorSize*sizeof(float));
     for(index = 0; index<VectorSize; index++)
       	 fscanf(fp, "%f", &Vector[index]);     

  }

  MPI_Bcast (&MatrixFileStatus, 1, MPI_INT, Root, MPI_COMM_WORLD);
  if(MatrixFileStatus == 0) {
     printf("Can't open input file for Matrix ..... \n");
     MPI_Finalize();
     exit(-1);
  }

  MPI_Bcast (&VectorFileStatus, 1, MPI_INT, Root, MPI_COMM_WORLD);
  if(VectorFileStatus == 0) {
     printf("Can't open input file for Vector ..... \n");
     MPI_Finalize();
     exit(-1);
  }

  if((VectorSize != NoofCols) || (NoofRows < Numprocs-1))
      ValidInput = 0;

  MPI_Bcast(&ValidInput, 1, MPI_INT, Root, MPI_COMM_WORLD);

  if(ValidInput  == 0){
     printf("Invalid input data..... \n");
     printf("NoofCols should be equal to VectorSize\n");
     MPI_Finalize();
     exit(0);
  }

  MPI_Bcast(&VectorSize, 1, MPI_INT, Root, MPI_COMM_WORLD);

  if(MyRank != 0) 
     Vector = (float *)malloc(VectorSize*sizeof(float));

  MPI_Bcast(Vector, VectorSize, MPI_FLOAT, Root, MPI_COMM_WORLD);

  /* .......Allocate memory for output and a row of a matrix......*/ 
  Buffer= (float *)malloc(NoofCols*sizeof(float));
  FinalVector = (float *)malloc(NoofRows*sizeof(float)); 
  for(irow=0; irow< NoofRows; irow++) 
      FinalVector[irow] = 0;

  /* First Send At least one row to each processors */
  for(irow = 1 ; irow < Numprocs ; irow++){
      for(icol=0; icol< NoofCols; icol++){	 
	  Buffer[icol] = Matrix[irow-1][icol];
      }
      MPI_Send( Buffer, NoofCols, MPI_FLOAT, irow, RowtoSend+1, 
		MPI_COMM_WORLD);
      RowtoSend++;
  }

  for(irow = 0 ; irow < NoofRows ; irow++){
      MPI_Recv(&Sum, 1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, 
	       MPI_COMM_WORLD, &status);
      Destination = status.MPI_SOURCE;
      Source_tag  = status.MPI_TAG;
      FinalVector[Source_tag] = Sum;

    /*For the  remaining rows(if any) send the data to whichever processor 
        that finishes the earlier computations first */

      if( RowtoSend < NoofRows ){
   	  Destination_tag = RowtoSend+1;
  	  for(icol=0; icol< NoofCols; icol++)
	      Buffer[icol] = Matrix[RowtoSend][icol];

   	  MPI_Send(Buffer, NoofCols, MPI_FLOAT, Destination, 
		   Destination_tag, MPI_COMM_WORLD);
	  RowtoSend++;
      }    
   }

   for(iproc = 1 ; iproc < Numprocs ; iproc++) 
       MPI_Send(Buffer, NoofCols, MPI_FLOAT, iproc, 0, MPI_COMM_WORLD); 
	
   for(irow = 0 ; irow < NoofRows ; irow++) 
       printf("FinalAnswer[%d] =  %f \n", irow, FinalVector[irow]);
		
   free(Vector);
   free(FinalVector);
   free(Buffer);

   MPI_Finalize(); 
}


