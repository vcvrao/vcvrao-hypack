/******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

    Example 2.1		: broadcast.c

    Objective           : To braodcast an integer array of size "n" by 
                          "with Rank 0"  using MPI Collective communication 
                           library call 

                          (MPI_Bcast)

                          This example demonstrates the use of
                          MPI_Init
                          MPI_Comm_rank
                          MPI_Comm_size
                          MPI_Bcast                      
                          MPI_Finalize

    Input               : Input Data file "sdata.inp" by proces with Rank 0

    Output              : Print the scattered array on all processes.

    Necessary Condition : Number of processes should be 
                          less than or equal to 8.

   Created              : August-2013

   E-mail               : hpcfte@cdac.in     

*******************************************************************/



#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include "mpi.h"
#define BUFFER_SIZE 14
int main(int argc,char **argv)
{
/*................VARIABLE INITIALIZATION...............*/
   int   MyRank,Numprocs;
   int   Root=0, Source=8;
   char message[BUFFER_SIZE];  


/*................MPI INITIALIZATION...............*/
   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);
   MPI_Comm_size(MPI_COMM_WORLD,&Numprocs);

/*..............GETTING MESSAGE...............*/

   if(MyRank==8)
   strcpy(message,"HELLO_WORLD");

 /*................CALL MPI BROADCAST..................*/

   MPI_Bcast(message,BUFFER_SIZE,MPI_CHAR,Source,MPI_COMM_WORLD);

/*...................PRINT RECIEVED MESSAGE..........*/

   printf("\n Rank %d gets %s from Process %d *** \n",MyRank,message,Root);  
         

 MPI_Finalize();
}
