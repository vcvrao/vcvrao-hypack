
/*******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example 4.8	      : multiple-communicators-cartesian.c 

  Objective           : Matrix Matrix multiplication
         		  (Using Cartesian Topology, CANNON Algorithm)

  Input               : Read files (mdata1.inp) for first input matrix and 
                        (mdata2.inp)for second input matrix 

  Output              : Result of matrix matrix multiplication on Processor 0.

  Necessary Condition : Number of Processes should be less than
                        or equal to 8. Matrices A and B should be
			equally striped. that is Row size and
			Column size should be properly divisible 
			by Number of processes used.

   Created            : August-2013

   E-mail             : hpcfte@cdac.in     

********************************************************************
*/



#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

//#include "cart.h"

/********************************cart.h******************************/
#define NO_OF_PROCESSES 4
#define MESH_SIZE 2
#define NDIMENSIONS 2 

typedef struct {
   int      Size;     /* The number of processors. (Size = q_proc*q_proc)
  */
   int      p_proc;        /* The number of processors in a row (column).
*/
   int      Row;      /* The mesh row this processor occupies.        */
   int      Col;      /* The mesh column this processor occupies.     */
   int      MyRank;     /* This processors unique identifier.           */
   MPI_Comm Comm;     /* Communicator for all processors in the mesh. */
   MPI_Comm Row_comm; /* All processors in this processors row   .    */
   MPI_Comm Col_comm; /* All processors in this processors column.    */
} MESH_INFO_TYPE;

/******************************cart.h**********************************/



/* Communication block set up for mesh toplogy */
void SetUp_Mesh(MESH_INFO_TYPE *);

main (int argc, char *argv[])
{

 int istage,irow,icol,jrow,jcol,iproc,jproc,index,Proc_Id,Root =0;
 int A_Bloc_MatrixSize, B_Bloc_MatrixSize;
 int NoofRows_A, NoofCols_A, NoofRows_B, NoofCols_B;
 int NoofRows_BlocA, NoofCols_BlocA, NoofRows_BlocB, NoofCols_BlocB;
 int Local_Index, Global_Row_Index, Global_Col_Index;
 int Matrix_Size[4];
 int source, destination, send_tag, recv_tag, Bcast_root;

 float **Matrix_A, **Matrix_B, **Matrix_C;
 float *A_Bloc_Matrix, *B_Bloc_Matrix, *C_Bloc_Matrix, *Temp_BufferA;
 float *MatA_array, *MatB_array, *MatC_array;
 
 FILE *fp;
 int     MatrixA_FileStatus = 1, MatrixB_FileStatus = 1;

 MESH_INFO_TYPE grid;
 MPI_Status status; 

 /* Initialising */
 MPI_Init (&argc, &argv);

 /* Set up the MPI_COMM_WORLD and CARTESIAN TOPOLOGY */
  SetUp_Mesh(&grid);
 
  MPI_Finalize();
 }

 /* Function : Finds communication information suitable to mesh topology  */
 /*            Create Cartesian topology in two dimnesions                */

 void SetUp_Mesh(MESH_INFO_TYPE *grid) {

   int Periods[2];      /* For Wraparound in each dimension.*/
   int Dimensions[2];   /* Number of processors in each dimension.*/
   int Coordinates[2];  /* processor Row and Column identification */
   int Remain_dims[2];      /* For row and column communicators */


 /* MPI rank and MPI size */
   MPI_Comm_size(MPI_COMM_WORLD, &(grid->Size));
   MPI_Comm_rank(MPI_COMM_WORLD, &(grid->MyRank));

 /* For square mesh */
   grid->p_proc = (int)sqrt((double) grid->Size);             
	if(grid->p_proc * grid->p_proc != grid->Size){
	 MPI_Finalize();
	 if(grid->MyRank == 0){
	 printf("Number of Processors should be perfect square\n");
	 }
		 exit(-1);
	}

   Dimensions[0] = Dimensions[1] = grid->p_proc;

   /* Wraparound mesh in both dimensions. */
   Periods[0] = Periods[1] = 1;    

   /*  Create Cartesian topology  in two dimnesions and  Cartesian 
       decomposition of the processes   */
   MPI_Cart_create(MPI_COMM_WORLD, NDIMENSIONS, Dimensions, Periods, 0, &(grid->Comm));
   MPI_Cart_coords(grid->Comm, grid->MyRank, NDIMENSIONS, Coordinates);

   grid->Row = Coordinates[0];
   grid->Col = Coordinates[1];

 /*Construction of row communicator and column communicators 
 (use cartesian row and columne machanism to get Row/Col Communicators)  */

   Remain_dims[0] = 0;            
   Remain_dims[1] = 1; 

 /*The output communicator represents the column containing the process */
   MPI_Cart_sub(grid->Comm, Remain_dims, &(grid->Row_comm));
   
   Remain_dims[0] = 1;
   Remain_dims[1] = 0;

 /*The output communicator represents the row containing the process */
   MPI_Cart_sub(grid->Comm, Remain_dims, &(grid->Col_comm));
}
