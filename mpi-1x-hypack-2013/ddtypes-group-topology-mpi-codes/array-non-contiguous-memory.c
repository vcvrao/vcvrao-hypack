/*
**********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

   Example 4.3	   : array-non-contiguous-memory.c

   Objective       : To build derived datatype in which process with 
                     rank 0 sends one row entries of a two dimensional 
                     real array, which are contiguous entries of two 
                     dimensional real array to the process with rank 1. 

   Input           : The input file for Two dimansioanl real array 

   Output          : Process with Rank 1 prints the elements of the 7th 
                     row of two dimensional real array 
	
   Created         : August-2013

   E-mail          : hpcfte@cdac.in     

**********************************************************************
*/

/*..........Program of MPI_Type_vector usage..................................*/

/*.Program: MPI parallel program to distribute the columns of the matrix among 
            the processes ............*/ 

#include<stdio.h>
#include<stdlib.h>
#include"mpi.h"
#define n_size 4

int main(int argc,char **argv)
{
  int Group_size,Rank,root=0,tag=0,i,iproc;
  MPI_Status status;
  int array[n_size][n_size];
  int row,col,array1[n_size][n_size],inc=0;
  MPI_Datatype new_col_type;
 
/*.........MPI initialization...............................*/                   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
   MPI_Comm_size(MPI_COMM_WORLD,&Group_size);

/*....Define the MPI Type Vector & commit it........................*/
   MPI_Type_vector(n_size,1,n_size,MPI_INT,&new_col_type);
   MPI_Type_commit(&new_col_type);


/*..........Read the input Matrix...................................*/


  if(Group_size == n_size)
  {
    if(Rank==0)
      {
         fflush(stdout);
         for(row=0;row<n_size;row++)
          {   
            for(col=0;col<n_size;col++)
            array[row][col]=row + (inc++); 
          }                                                               
 
/*............Print the original input matrix.................*/ 
     printf("The orignal Matrix\n"); 
     for(row=0;row<n_size;row++)
      {   
        for(col=0;col<n_size;col++)
        	printf("%d\t",array[row][col]); 
       printf("\n"); 
      }

/*....send the column of matrix to the proceses according to Rank....*/

   for(iproc=1;iproc<Group_size;iproc++)
      MPI_Send(&array[0][iproc],1,new_col_type,iproc,tag,MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD); 
 
/*...Receive the column of matrix & display the output..............*/
  if(Rank != 0)
  {
  	MPI_Recv(&array1[0][Rank],1,new_col_type,0,tag,MPI_COMM_WORLD,&status);
 	printf(" Myrank is %d & recived col is %d \n",Rank,Rank);
  	for(i=0;i<n_size;i++)
  		printf("%d\n",array1[i][Rank]);
  }
}

/*.....If the No. of process is not equal to no. of column..............*/ 
 else
  {
    if(Rank==0)
    printf("\n You should have %d Processes to run this.Terminated\n",n_size);
  }
  
/*.....MPI finalizing................................*/
 MPI_Finalize();
 exit(0);
}
  
