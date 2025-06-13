/*
*****************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

   Example 5.4:	       : mat_infnorm_blkstp.c

   Objective           : Calculate Infinity Norm using block striped 
                         partitioning

   Input               : Read file (infndata.inp) for Matrix

   Output              : Process with rank 0 prints the value of Infinity 
                         Norm 

   Necessary Condition : Number of Processes should be
                         less than or equal to 8

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

*****************************************************************/

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

 main(int argc, char** argv) 
 {

  int	   Numprocs, MyRank;
  int 	NoofCols, NoofRows, ScatterSize; 
  int	   index, irow, icol;
  int	   Root = 0;
  float  **InputMatrix, *Buffer, *MyBuffer;
  float	max=0, sum=0, Inf_norm=0;
  FILE	*fp;
  int		MatrixFileStatus = 1;

/* ........MPI Initialisation .......*/

  MPI_Init(&argc, &argv); 
  MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
  MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);
	
  if(MyRank == 0) {
	
/*.......Read the Matrix Input file ......*/
  if((fp = fopen ("./data/infndata.inp", "r")) == NULL){
	MatrixFileStatus = 0;
  }

  if( MatrixFileStatus != 0 ) {
     fscanf(fp, "%d %d\n", &NoofRows, &NoofCols);     

/* .......Allocate memory for matrix .....*/
     InputMatrix  = (float **) malloc(NoofRows*sizeof(float *));
     for(irow=0; irow<NoofRows; irow++)
     InputMatrix[irow] = (float *) malloc(NoofCols*sizeof(float));

/* .......Read data for matrix .....*/
    for(irow=0; irow< NoofRows; irow++) {
   	for(icol=0; icol< NoofCols; icol++) 
     	  fscanf(fp, "%f", &InputMatrix[irow][icol]); 
    }
 fclose(fp); 
       
/* .......Convert 2-D InputMatrix into 1-D Array .....*/
     Buffer = (float *)malloc(NoofRows*NoofCols*sizeof(float));

     index = 0;
    for(irow=0; irow< NoofRows; irow++){
  	for(icol=0; icol< NoofCols; icol++) {
     	  Buffer[index] = InputMatrix[irow][icol]; 
          index++;
	}
    }
  }

 }/* MyRank == 0 */


   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Bcast(&MatrixFileStatus, 1, MPI_INT, Root, MPI_COMM_WORLD);
	if(MatrixFileStatus == 0) {
     	  if(MyRank == Root) printf("Can't open Matrix input file");
	  MPI_Finalize();
     	  exit(-1);
	}

   MPI_Bcast(&NoofRows, 1, MPI_INT, Root, MPI_COMM_WORLD);

   if(NoofRows < Numprocs) {
     MPI_Finalize();
     if(MyRank == 0)
     printf("Noof Rows should be more than No of Processors ... \n");
     exit(0);
   }  	

   if(NoofRows % Numprocs != 0) {
     MPI_Finalize();
     if(MyRank == 0) {
        printf("Matrix Can not be Striped Evenly ..... \n");
      }
   exit(0);
   }  	

   MPI_Bcast(&NoofCols, 1, MPI_INT, Root, MPI_COMM_WORLD);

   ScatterSize = NoofRows / Numprocs;
   MyBuffer = (float *)malloc(ScatterSize * NoofCols * sizeof(float));
   MPI_Scatter( Buffer, ScatterSize * NoofCols, MPI_FLOAT, 
		 MyBuffer, ScatterSize * NoofCols, MPI_FLOAT, 
					 0, MPI_COMM_WORLD);

 max=0;	
 for(irow=0;irow<ScatterSize;irow++) {
	sum = 0;
	index = irow*NoofCols;
  for(icol=0;icol<NoofCols;icol++) {
   sum += (MyBuffer[index]>=0)?(MyBuffer[index]):(0-MyBuffer[index]); 
   index++;
  }
  max = max<sum ? sum : max;
 }

MPI_Reduce(&max,&Inf_norm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
 if(MyRank==0){
	max=0;
/* Serial Check */
  for(irow=0;irow<NoofRows;irow++) {
    sum = 0;
    index = irow*NoofCols;
   	for(icol=0;icol<NoofCols;icol++) {
	sum += (Buffer[index]>=0)?(Buffer[index]):(0-Buffer[index]); 
	index++;
	}
    max = max<sum ? sum : max;
   }

printf("The Infinity Norm Is(Parallel Code) : %f\n",Inf_norm);
printf("The Infinity Norm Is(Serial Code)   : %f\n",max);

 }
 MPI_Finalize();
 }



