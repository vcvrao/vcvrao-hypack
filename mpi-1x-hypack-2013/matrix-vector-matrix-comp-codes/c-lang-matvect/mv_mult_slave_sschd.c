/*
********************************************************************	

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

   Example 5.5	       : mv_mult_slave_sschd.c

   Objective           : Matrix_Vector Multiplication
                         (Self_scheduling algorithm worker Program)

   Input               : A matrix row from master 

   Output              : Computed result to master 

   Necessary Condition : Number of processes should be greater than 
                         2 and less than or equal to 8

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

******************************************************************/




#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

main(int argc, char** argv) 
{

	int		  Numprocs, MyRank;
	int		  Source, Destination;
	int		  Source_tag, Destination_tag;
	float		  *MyBuffer, *Vector, Sum;
	int		  VectorSize;
  	int 		  index, flag, ValidInput = 1;
	int	     Root = 0;
	int		  MatrixFileStatus = 1, VectorFileStatus = 1;
  	MPI_Status status;     

   /* ........MPI Initialisation .......*/
  	MPI_Init(&argc, &argv); 
  	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
  	MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);

	if(Numprocs < 2) {
	   MPI_Finalize();
	   exit(0);
	}  

   MPI_Bcast (&MatrixFileStatus, 1, MPI_INT, Root, MPI_COMM_WORLD);
	if(MatrixFileStatus == 0) {
	   MPI_Finalize();
           exit(-1);
	}

   MPI_Bcast (&VectorFileStatus, 1, MPI_INT, Root, MPI_COMM_WORLD);
	if(VectorFileStatus == 0) {
	   MPI_Finalize();
           exit(-1);
	}

        MPI_Bcast(&ValidInput, 1, MPI_INT, Root, MPI_COMM_WORLD);

	if(ValidInput==0){
           MPI_Finalize();
	   exit(0);
	}

	MPI_Bcast(&VectorSize, 1, MPI_INT, 0, MPI_COMM_WORLD); 

        Vector = (float *)malloc(VectorSize*sizeof(float));   
	MPI_Bcast (Vector, VectorSize, MPI_FLOAT, 0, MPI_COMM_WORLD); 

        MyBuffer = (float *)malloc(VectorSize*sizeof(float));   

        for(;;) {
	   MPI_Recv(MyBuffer, VectorSize, MPI_FLOAT, Root, MPI_ANY_TAG, 
					 MPI_COMM_WORLD, &status);
           flag = status.MPI_TAG - 1;

	   if(flag == -1) 
	      break;

	   Destination = 0;
	   Destination_tag = flag;

	   Sum = 0.0;
	   for(index=0; index< VectorSize; index++) 
	       Sum += (MyBuffer[index] * Vector[index]);
	   MPI_Send(&Sum, 1, MPI_FLOAT, 0, Destination_tag,MPI_COMM_WORLD);
        }

	free(MyBuffer);
	free(Vector);

        MPI_Finalize();
}



