
/*
***********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

   Example 1.10		: sum_associative_fanin_nonblocking_tree.c

   Objective           : To find sum of 'n' integers on 'p' processors using 
                         'Associative Fan-in' rule.
                         MPI Non-Blocking communication library calls are used.

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

   Necessary           : Number of Processes should be less than 
   Condtion              or equal to 8.

   Created             : August-2013

   E-mail              : hpcfte@cdac.in     


************************************************************************
*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define  LOG2(x)    log10((double)(x)) / log10((double) 2)
#define  IS_INT(x)  ((x) == (int)(x))?1:0

int main(int argc,char *argv[])
{
    int	   MyRank, Numprocs;
    int    Root = 0;
    int    ilevel, value, ans;
    int	   Source, Source_tag;
    int    Destination, Destination_tag;
    int	   Level, NextLevel;
    float  NoofLevels;
    static int	 sum = 0;
    MPI_Status  *status;
    MPI_Request request[200];


    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&Numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyRank);

    NoofLevels = LOG2(Numprocs);
    if(!(IS_INT(NoofLevels))){
        if(MyRank == Root)
	   printf("\nNumber of processors should be power of 2\n");
        MPI_Finalize();
        exit(-1);
    }

    sum = MyRank; 
    Source_tag = 0;
    Destination_tag = 0; 

    for(ilevel = 0 ; ilevel  < NoofLevels; ilevel ++){
	Level = (int)(pow((double)2, (double)ilevel ));
	
	if((MyRank % Level) == 0){
	    NextLevel = (int)(pow((double)2, (double)(ilevel +1)));

	    if((MyRank % NextLevel) == 0){
	        Source = MyRank + Level;
	        MPI_Irecv(&value, 1, MPI_INT, Source, Source_tag, 
			   MPI_COMM_WORLD, &request[ilevel]);
  	        MPI_Waitall(1,&request[ilevel],status); 
                sum = sum + value;
	    }
	    else{
	        Destination = (MyRank - Level);
	        MPI_Isend(&sum, 1, MPI_INT, Destination, Destination_tag, 
			   MPI_COMM_WORLD,&request[ilevel]);
  	     	MPI_Waitall(1,&request[ilevel],status); 
	    }
        }
    }
	     
    if(MyRank == Root) 
       printf(" My Rank is %d Final SUM is%d\n", MyRank, sum);

    MPI_Finalize();
}



