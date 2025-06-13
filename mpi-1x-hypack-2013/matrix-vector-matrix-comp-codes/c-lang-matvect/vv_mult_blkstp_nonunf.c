
/*
********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

   Example 5.2		: vv_mult_blkstp_nonunf.c

   Objective           : Vector_Vector Multiplication 
                         (Using Non uniform Data Partitioning)

   Input               : Read files (vdata1.inp) for Vector_A and 
                         (vdata2.inp) for Vector_B 

   Output              : Result of Vector-Vector Multiplication on process 0

   Necessary Condition : Number of Processes should be
                         less than or equal to 8

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

******************************************************************/

#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#define Epsilon 1.0E-10
main(int argc, char** argv) 
{

  int	   Numprocs, MyRank;
  int 	VectorSize, VectorSize_A, VectorSize_B; 
  int	   index, iproc, jproc, disp_value;
  int	   Root = 0, ValidInput = 1, ValidOutput = 1, DataSize;
  int    Distribute_Cols, Remaining_Cols, *SendCount, *Displacement;
  int    Destination, Destination_tag = 0, Source, Source_tag = 0;
  int    RecvCount;
  float  *Mybuffer_A, *Mybuffer_B,  MyFinalVector, FinalAnswer;
  float  CheckResultVector;
  FILE	*fp;
  int		VectorA_FileStatus = 1, VectorB_FileStatus = 1; 
  float  *Vector_A, *Vector_B;
  MPI_Status status;


/* ........MPI Initialisation .......*/

  MPI_Init(&argc, &argv); 
  MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
  MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);

  if(MyRank == Root) {

/*.......Read the Vector A Input file ......*/
	  
     if((fp = fopen ("./data/vdata1.inp", "r")) == NULL) {
	     VectorA_FileStatus = 0;
  	  }

	  if(VectorA_FileStatus != 0) {
  	  	fscanf(fp, "%d\n", &VectorSize_A);     

/* .......Allocate memory and read data for vector A .....*/

          Vector_A  = (float *) malloc(VectorSize_A*sizeof(float));

/* .......Read data for matrix .....*/

     		for(index=0; index< VectorSize_A; index++)  {
     		  fscanf(fp, "%f", &Vector_A[index]); 
     		}
     		fclose(fp); 
     }

/*.......Read the Vector B Input file ......*/
	 
     if((fp = fopen ("./data/vdata2.inp", "r")) == NULL) {
	     VectorB_FileStatus = 0;
     }

	  if(VectorB_FileStatus != 0) {
     		fscanf(fp, "%d\n", &VectorSize_B);
	
/* .......Allocate memory and read data for vector B .....*/

            Vector_B = (float *)malloc(VectorSize_B*sizeof(float));
     	      for(index=0; index< VectorSize_B; index++) {
        	fscanf(fp, "%f", &Vector_B[index]);
     		}
    fclose(fp);
	    }

	  if(VectorSize_A != VectorSize_B)
	     ValidInput = 0;		  

	}
/* MyRank == 0 */

MPI_Barrier(MPI_COMM_WORLD);

MPI_Bcast(&VectorA_FileStatus, 1, MPI_INT, Root, MPI_COMM_WORLD);
 if(VectorA_FileStatus == 0) {
   if(MyRank == Root) printf("Can't open input file for Vector A\n");
   MPI_Finalize();
   exit(-1);
 }

MPI_Bcast(&VectorB_FileStatus, 1, MPI_INT, Root, MPI_COMM_WORLD);
   if(VectorB_FileStatus == 0) {
    if(MyRank == Root) printf("Can't open input file for Vector B\n");
    MPI_Finalize();
    exit(-1);
   }

MPI_Bcast(&ValidInput, 1, MPI_INT, Root, MPI_COMM_WORLD);

 if(ValidInput == 0) {
   if(MyRank == Root) printf("Size of both vectors is not equal .\n");
   MPI_Finalize();
   exit(-1);
   }

VectorSize = VectorSize_A;
MPI_Bcast(&VectorSize, 1, MPI_INT, Root, MPI_COMM_WORLD);

   if(VectorSize < Numprocs) {
     MPI_Finalize();
     if(MyRank == 0)
      printf("VectorSize should be more than No of Processors . \n");
     exit(0);
    }  	

/* Initial arrangement for Scatterv operation */

  if(MyRank == Root) {
     Displacement = (int *)malloc(Numprocs*sizeof(int));
     SendCount    = (int *)malloc(Numprocs*sizeof(int));

	Distribute_Cols = VectorSize/Numprocs;
	Remaining_Cols  = VectorSize % Numprocs;
	for(iproc = 0; iproc < Numprocs; iproc++)
          SendCount[iproc] = Distribute_Cols;

	for(iproc=Remaining_Cols; iproc>0; iproc--)
	  (SendCount[Remaining_Cols-iproc])++ ;

	Displacement[0] = 0;
	for(iproc = 1; iproc < Numprocs; iproc++) {
	  disp_value = 0;
	  for(jproc = 0; jproc < iproc; jproc++)
	     disp_value += SendCount[jproc];
	  Displacement[iproc] = disp_value;
        }

/* Send RecvCount to each process */

	for(iproc = 1; iproc < Numprocs; iproc++) {
	  Destination = iproc;
	  DataSize = 1;
	  RecvCount = SendCount[iproc];
          MPI_Send(&RecvCount, DataSize, MPI_INT, Destination, 
	  Destination_tag,MPI_COMM_WORLD);
        }
        RecvCount = SendCount[0];
   }

 if(MyRank != Root) {
   Source = Root;
   DataSize = 1;
   MPI_Recv(&RecvCount, DataSize, MPI_INT, Source, Source_tag, MPI_COMM_WORLD, &status);
	}

	/* Scatter Vector A and Vector B */

   Mybuffer_A = (float *)malloc(RecvCount * sizeof(float));
	MPI_Scatterv( Vector_A, SendCount, Displacement, MPI_FLOAT,Mybuffer_A, RecvCount, MPI_FLOAT, 
					 Root, MPI_COMM_WORLD);

   Mybuffer_B = (float *)malloc(RecvCount * sizeof(float));
	MPI_Scatterv( Vector_B, SendCount, Displacement, MPI_FLOAT,Mybuffer_B, RecvCount, MPI_FLOAT, 
					 Root, MPI_COMM_WORLD);

	/* Calculate partial sum */

	MyFinalVector = 0.0;
	for(index = 0 ; index < RecvCount ; index++) 
		 MyFinalVector += (Mybuffer_A[index] * Mybuffer_B[index]);

	/* Gather final answer on process 0 */

	DataSize = 1;
   MPI_Reduce(&MyFinalVector, &FinalAnswer, DataSize, MPI_FLOAT, MPI_SUM, Root, MPI_COMM_WORLD);

	if(MyRank == 0){
	   CheckResultVector = 0.0;
		for(index = 0 ; index < VectorSize ; index++) 
		 CheckResultVector += (Vector_A[index] * Vector_B[index]);
		if(fabs((double)(FinalAnswer-CheckResultVector)) > Epsilon){
		   printf("Error %d\n",index);
			ValidOutput = 0;
		}
		if(ValidOutput)
		  printf("FinalAnswer = %f\n", FinalAnswer);
        else
		  printf("Result may be wrong\n");
    
      free(Displacement);
      free(SendCount);
      free(Vector_A);
      free(Vector_B);
	 
   }
  
   free(Mybuffer_A);
   free(Mybuffer_B);

   MPI_Finalize();
}




