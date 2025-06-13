/*
**********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

   Example 4.2	 : array-contiguous-memory.c

   Objective     : To build derived datatype in which process with 
                   rank 0 sends one row entries of a two dimensional real array, 
                   which  contiguous entries of two dimensional real array to 
                   the process with rank 1. 

   Input         : The input file for Two dimansioanl real array 


   Output        : Process with Rank 1 prints the elements of the 7th row of 
                   two dimensional real array 

   Created       : August-2013

   E-mail        : hpcfte@cdac.in     

**********************************************************************
*/



#include<stdio.h>
#include"mpi.h"

int main(int argc, char *argv[]) 
{
   int rank;
   MPI_Status status;
   struct{
    int x;
    int y;
    int z;
   }point;
   MPI_Datatype ptype;

/*...............MPI Initialization.............*/   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&rank);

/*..............Use of MPI_Type_contiguous......*/ 
   MPI_Type_contiguous(3,MPI_INT,&ptype);
   MPI_Type_commit(&ptype);

/*.............Process with Rank 0 will send data to process 1......*/
   if(rank==0)
   {
      point.x=10; point.y=20; point.z=30;
      MPI_Send(&point,1,ptype,1,52,MPI_COMM_WORLD);
   } 

/*............Process with Rank 1 will receive the values and print the result....*/
   else if(rank==1) 
   {
      MPI_Recv(&point,1,ptype,0,52,MPI_COMM_WORLD,&status);
      printf("P:%d received coords are (%d,%d,%d) \n",rank,point.x,point.y,point.z);
   }

/*...........MPI Finalizing............*/ 
   MPI_Finalize();

   return 0;
}
