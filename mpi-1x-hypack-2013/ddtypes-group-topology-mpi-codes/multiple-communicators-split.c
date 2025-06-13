
/*
**********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

   Example 4.7	       : multiple-communicators-split.c

   Objective           : To create a diagonal communicator group of processors 
                         in a square grid processes on cluster

   Input               : None  

   Output              : Each Process print the Rank in old and new communicator 

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

*********************************************************************
*/

 #include<stdio.h>
 #include"mpi.h"
 int main(int argc,char **argv)
 {

   int Myrank,Numprocs,col=2,row=3,irow,jcol,row_id,col_id;
   MPI_Comm row_comm,col_comm;

/*.....MPI Initialization.....................................*/
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&Myrank);
  MPI_Comm_size(MPI_COMM_WORLD,&Numprocs);

/*....If No. of process is not equals to 2....................*/

  if(Numprocs != 6)
  {
    if(Myrank == 0)
      printf("\n Use 6 processes to run this program.terminated\n\n");
   MPI_Finalize();
   exit(0);
  } 

/*....Compute color & Key value for MPI_comm_split.............*/
  irow = Myrank / col;
  jcol = Myrank % col;

/*..Call MPI_Comm_split to creat new communicator group.........*/
  MPI_Comm_split(MPI_COMM_WORLD,irow,jcol,&row_comm);
  MPI_Comm_split(MPI_COMM_WORLD,jcol,irow,&col_comm);

/*...get rank in new communicator...............................*/   
  MPI_Comm_rank(row_comm,&row_id);
  MPI_Comm_rank(col_comm,&col_id); 

/*....Print the output..........................................*/
  if(Myrank==0)
  {
    printf("\n................Program of MPI_Comm_split....................\n");
    printf("\n..Split 3x2 grid into 2 different communicators having 3 rows & 2 columns.......\n");
 
    printf("\n    Myrank    irow    jcol    row_id     col_id\n");

  }

  MPI_Barrier(MPI_COMM_WORLD);
  printf("\n %8d %8d %8d %8d %8d \n",Myrank,irow,jcol,row_id,col_id );


/*...MPI finalizing,,,,,,,,,,,,,,,,,*/
MPI_Finalize();
 
}
