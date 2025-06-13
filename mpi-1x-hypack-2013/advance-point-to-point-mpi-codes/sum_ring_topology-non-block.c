/*
**********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example 3.1	    : sum_ring_topology-non-block.c

  Objective           : To find sum of 'n' integers on 'p' processors using 
                        Point-to-Point communication library calls and ring 
                        topology.MPI Non-Blocking communication library calls
                        are used.

                        This example demonstrates the use of 
                        MPI_Init
                        MPI_Comm_rank
                        MPI_Comm_size
                        MPI_Isend
                        MPI_Irecv
                        MPI_Waitall
                        MPI_Finalize

  Input               : Automatic input generation
                        The rank of each process is input on each process.

  Output              : Process with Rank 0 should print the sum of 'n' values 


  Necessary Condition : Number of Processes should be less than 
                        or equal to 8.

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     

***********************************************************************
*/

 

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc,char *argv[])
{
    int         MyRank, Numprocs, Root = 0;
    int         value, sum = 0;
    int	        Source, Source_tag;
    int         Destination, Destination_tag;
  
    //MPI_Status  *status;
    MPI_Status  status[200];
    MPI_Request request[200];

     /* Initialize MPI */
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&Numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);
   
    if (MyRank == Root){
	Destination     = MyRank + 1;
	Destination_tag = 0;
 
        MPI_Isend(&MyRank, 1, MPI_INT, Destination, Destination_tag, 
		 MPI_COMM_WORLD,&request[Destination]);
  	MPI_Waitall(1,&request[Destination],status); 

	/* MPI_Send(&MyRank, 1, MPI_INT, Destination, Destination_tag, 
		  MPI_COMM_WORLD); */
    }

    else{
       if(MyRank < Numprocs-1 ){
	  Source     = MyRank - 1;
	  Source_tag = 0;

	   MPI_Irecv(&value, 1, MPI_INT, Source, Source_tag, 
			   MPI_COMM_WORLD, &request[Source]);
  	   MPI_Waitall(1,&request[Source],status); 
          

	   /* MPI_Recv(&value, 1, MPI_INT, Source, Source_tag, 
		    MPI_COMM_WORLD, &status); */

	  sum  = MyRank + value;
	  Destination = MyRank + 1;
	  Destination_tag = 0;

    	  MPI_Isend(&MyRank, 1, MPI_INT, Destination, Destination_tag, 
		 MPI_COMM_WORLD,&request[Destination]);

  	  MPI_Waitall(1,&request[Destination],status); 

          /* MPI_Send(&sum, 1, MPI_INT, Destination, Destination_tag, 
		    MPI_COMM_WORLD); */

	}
	else{
	   Source     = MyRank - 1;
	   Source_tag = 0;


	        MPI_Irecv(&value, 1, MPI_INT, Source, Source_tag, 
			   MPI_COMM_WORLD, &request[Source]);
  	        MPI_Waitall(1,&request[Source],status); 

	       /*  MPI_Recv(&value, 1, MPI_INT, Source, Source_tag, 
		     MPI_COMM_WORLD, &status); */

	       sum = MyRank + value;
	}
    } 
    
    if (MyRank == Root)
    {
	Source     = Numprocs - 1;
	Source_tag = 0;

	MPI_Irecv(&sum, 1, MPI_INT, Source, Source_tag, 
			   MPI_COMM_WORLD, &request[Source]);
  	MPI_Waitall(1,&request[Source],status); 
             

	/* MPI_Recv(&sum, 1, MPI_INT, Source, Source_tag, 
		  MPI_COMM_WORLD, &status); */

	printf("MyRank %d Final SUM %d\n", MyRank, sum);

    }

    if(MyRank == (Numprocs - 1)){
       Destination     = 0;
       Destination_tag = 0;
      

        MPI_Isend(&sum, 1, MPI_INT, Destination, Destination_tag, 
		 MPI_COMM_WORLD,&request[Destination]);

        MPI_Waitall(1,&request[Destination],status); 
 
       /*  MPI_Send(&sum, 1, MPI_INT, Destination, Destination_tag, 
		  MPI_COMM_WORLD); */

    }

    MPI_Finalize();

}




