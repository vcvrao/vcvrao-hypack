/*

 *	C-DAC Tech Workshop : hyPACK-2013
 *               October 15-18, 2013
 *
 * Program : mpi_tbb_matrix_vector_multiply.cpp
 * Desc.   : [MPI + IntelTBB] program to do vector matrix multiplication.
 * Input   : None
 * Output  : Resultant matrix after matrix-vector multiplication
 *
 * Email   : hpcfte@cdac.in
 *
*/

#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<assert.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

using namespace tbb;
using namespace std;




/*
 * Class construct 'PartialVectorCalc' is used by intelTBB's parallel_for API call.
 * This API call executes the for-loop in parallel. All the details of this operation
 * is taken care of by intelTBB API.
 */
class PartialVectorCalc
{
  private:
    float *const LocalMatrix, *const Vector, *const ResultVector;//private members of this class
    int *const NumColumns;
  public:

    //constructor initializing private members to parameter-passed.
    PartialVectorCalc(float *A, float *B, float *R, int *NC):LocalMatrix(A), Vector(B), ResultVector(R), NumColumns(NC){}
	
    //operator overloading ; required by intelTBB API
    void operator()(blocked_range<size_t>&r) const
    {
      float *vA, *vB, *vR;
      vA = LocalMatrix;
      vB = Vector;
      vR = ResultVector;
      int *NumCol = NumColumns;
      int index=0;
      
      
       for(size_t i=r.begin(); i!=r.end(); ++i)
      {
	vR[i]=0;
	index = i * (*NumCol);//shifts to 'i'th row of vA (i.e., Local Matrix)

	for(size_t j=0; j<(*NumCol); j++)
	{
	  vR[i] += (vA[index++] * vB[j]);	
	}
      }      
    }
};//end of class PartialVectorCalc 



int main(int argc, char * argv[])
{
  int ProcessId;//Id of the process (rank)
  int NumProcs;//total no. of processes
  
  FILE* MatrixFileDesc;//file descriptor of Matrix input file
  FILE* VectorFileDesc;//file descriptor of Vector input file
  int FileSuccess = 1;//file access success indicator (default to 'true')
  int iRow, iCol, i;//temps
  
  float **Matrix;//Pointer to Matrix
  float *Vector;//Pointer to Vector
  float *Buffer;//Stores 1D Matrix
  float *LocalMatrix;//Part of Matrix with each process in 1D
  float *LocalResultVec;//Local partial result of the vector
  float *FinalResultVec;//Final result vector
  
  int NumColumns;//no. of columns in Matrix
  int NumRows;//no. of rows in Matrix
  int VectorSize;//size of Vector
  
  int NumRowsInPartn;//number of rows in each partition
  int PartnSize;//size of each partition

  int TermFlag=0;
  
  //initialize mpi communicator
  MPI_Init(&argc, &argv);
  
  //get this process id
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcessId);

  //get total processes
  MPI_Comm_size(MPI_COMM_WORLD, &NumProcs);
  
  /*
   * Operation exclusive to Rank=0:
   * Read: input Matrix from file "Matrix.input" and
   * 	   input Vector from "Vector.input"
   * Convert 2D 'Matrix' to 1D 'Buffer'
   */
  if(ProcessId == 0)//if process rank is '0'
  {
    
    
    //read Matrix from "./Matrix.input"
    if((MatrixFileDesc = fopen("./Matrix.input", "r"))==NULL)
	cout << "\nError opening Matrix.input\n";
    if(MatrixFileDesc != NULL)//if no error opening
    {
      //get NumRows & NumColumns
      fscanf(MatrixFileDesc, "%d %d\n", &NumRows, &NumColumns);
	
      //allocate memory and read Matrix
      Matrix = (float**)malloc(NumRows * sizeof(float*));
      
      //read all values of Matrix[NumRows][NumColumns] from "Matrix.input"
      for(iRow=0; iRow<NumRows; iRow++)
      {
	Matrix[iRow] = (float*) malloc(NumColumns * sizeof(float));
	
	for(iCol=0; iCol<NumColumns; iCol++)
	{
	  fscanf(MatrixFileDesc, "%f", &Matrix[iRow][iCol]);
	}
      }
      //close "Matrix.input" file
      fclose(MatrixFileDesc);
      
      //convert Matrix from 2D to 1D and store in Buffer[NumRows*NumColumns]
      Buffer = (float*) malloc(NumRows * NumColumns * sizeof(float));
      i=0;
      for(iRow=0; iRow<NumRows; iRow++)
	for(iCol=0; iCol<NumColumns; iCol++)
	  Buffer[i++] = Matrix[iRow][iCol];
      
    }
    else//if error opening "Matrix.input"
    {
      FileSuccess = 0;//Unsuccessful file operation
    }
    
    
    //read Vector from "./Vector.input"
    if((VectorFileDesc = fopen("./Vector.input", "r"))==NULL)
	cout << "\nError opening Vector.input\n";
	
    if(VectorFileDesc != NULL)//if no error opening
    {
      //get VectorSize
      fscanf(VectorFileDesc,"%d\n", &VectorSize);

      //allocate memory and read Vector
      Vector = (float*) malloc(VectorSize * sizeof(float));

      //read all values of Vector[VectorSize] from "Vector.input"
      for(iRow=0; iRow<VectorSize; iRow++)
      {
	fscanf(VectorFileDesc,"%f", &Vector[iRow]);
      }
      
      //close "Vector.input" file
      fclose(VectorFileDesc);
    }
    else//if error opening "Vector.input"
    {
      FileSuccess = 0;//Unsuccessful file operation    
    }

  }
  
  
  //barrier synchronization; wait for all tasks to reach here
  MPI_Barrier(MPI_COMM_WORLD);

  //Broadcast all FileSuccess
  MPI_Bcast(&FileSuccess, 1, MPI_INT, 0, MPI_COMM_WORLD);//Root = 0
  if(FileSuccess == 0)//if either Matrix.input/Vector.input failed to open
  {
    //all processes exit 'gracefully'
    if(ProcessId == 0)
    {
	printf("Couldn't open either or both data input files\n");
    }
    //end MPI communicator
    MPI_Finalize();
    exit(1);//exit program with error
  }
  
  //Braodcast No. of Rows
  MPI_Bcast(&NumRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  //Braodcast No. of Columns
  MPI_Bcast(&NumColumns, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  //Braodcast VectorSize
  MPI_Bcast(&VectorSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
 
  //no. of partitions obtained
  NumRowsInPartn = NumRows / NumProcs;
  
  //if no. of rows is zero or not a multiple of total-processes or NumColumns != VectorSize
  if(NumRows==0 || NumRows%NumProcs != 0 || NumColumns != VectorSize )
  {
    if(ProcessId == 0)
    {
      if(NumRows%NumProcs != 0)
	printf("Number of rows not a multiple of total processes\n");
      if(NumColumns != VectorSize)
	printf("Number of columns not equal to vector size\n");
      if(NumRows==0)
	printf("Number of rows is equal to zero!\n");
    }
    //all processes exit gracefully
    MPI_Finalize();
    exit(1);
  }
  //broadcast Vector to all processors
  if(ProcessId != 0)//if not process with rank '0' (root)
  {
    Vector = (float*)malloc(VectorSize * sizeof(float));//allocate memory
  }
  MPI_Bcast(Vector, VectorSize, MPI_FLOAT, 0, MPI_COMM_WORLD); 
  
  PartnSize = (NumColumns * NumRows) / NumProcs;//size of each partition
  LocalMatrix = (float*) malloc((PartnSize) * sizeof(float));//allocate memory for LocalMatrix
  
  //scatter Buffer(1D Matrix) to all processes and store in LocalMatrix
  MPI_Scatter(Buffer, PartnSize, MPI_FLOAT, LocalMatrix, PartnSize, MPI_FLOAT,0, MPI_COMM_WORLD);
 
  
LocalResultVec = (float*)malloc(NumRowsInPartn * sizeof(float));
  
  //initialize LocalResultVec
  for(iRow=0; iRow<NumRowsInPartn; iRow++)
    LocalResultVec[iRow] = 0.0;
   
  
  
  //initialize TBB task scheduler
  task_scheduler_init init;
  
  //barrier synchronization; wait for all tasks to reach here
  MPI_Barrier(MPI_COMM_WORLD);
  
  //TBB parallel_for; parallel execution of matrix-vector multiplication on shared-memory-processor(SMP)
  //parallel_for(blocked_range<size_t>(0, NumRowsInPartn, 1000), PartialVectorCalc(LocalMatrix, Vector, LocalResultVec, &NumColumns));
  parallel_for(blocked_range<size_t>(0, NumRowsInPartn), PartialVectorCalc(LocalMatrix, Vector, LocalResultVec, &NumColumns));
  
  

//par_matrix_vector_multiply( NumRowsInPartn,NumColumns,LocalMatrix,Vector,LocalResultVec);
  //par_matrix_vector_multiply( NumRowsInPartn,NumColumns,LocalMatrix,Vector,LocalResultVec);
  //par_matrix_vector_multiply( PartnSize,NumColumns,LocalMatrix,Vector,LocalResultVec);
	
  //barrier synchronization; wait for all tasks to reach here
  MPI_Barrier(MPI_COMM_WORLD);
  //allocate memory for FinalResultVec
  FinalResultVec = (float*) malloc(NumColumns * sizeof(float));
  

  //Gather all partial LocalResultVec from each MPI process to FinalResultVec
  MPI_Gather(LocalResultVec, NumRowsInPartn, MPI_FLOAT, FinalResultVec, NumRowsInPartn, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  if(ProcessId == 0)//if process '0' (root)
  {
    //display FinalResultVec
    for(iRow=0; iRow<NumColumns; iRow++)
      printf("%f\n", FinalResultVec[iRow]);
  
  }
  // added later 
  if(ProcessId==0)
  {
  free(FinalResultVec);
  free(Buffer);
  }
 
  free(Vector);
  free(LocalMatrix);
  free(LocalResultVec);
  //exit MPI processes  
  MPI_Finalize();

  return 0;
}
