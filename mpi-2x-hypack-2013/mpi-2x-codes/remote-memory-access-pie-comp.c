/*
 * ***********************************************************************
 *
 *		C-DAC Tech Workshop : hyPACK-2013
 *                           October 15-18, 2013
 *
 *                       Example 1.4 (Pie computation - Using RMA winwdows) 
 *
 * Objective            : Compute Pie value use MPI 2.0 Library Calls 
 *
 * Input                : None 
 *
 * Description          : Use MPI RMA Windows library calls & Calculate Pie Value
 *                        Based on the number of processes spawned the program
 *                        Each process computes its partial sum of numerical 
 *                        integration of "pie" value & accumalates final result                    
 *
 * Output               : Reads a file  
 *
 * Necessary conditions : Number of Processes should be less than or equal to 8
 *
 *  Created             : August-2013
 *
 *  E-mail              : hpcfte@cdac.in     
 *
 ***********************************************************************
 */


/* Computer pi by numerical integration, RMA version */

#include "mpi.h"
#include <stdio.h>
#include<stdlib.h>

#include <math.h>

int main(int argc, char *argv[])
{
	int n, myid, numprocs, i;
	double PI25DT = 3.141592653589793238462643;
	double mypi, pi, h, sum, x;
	MPI_Win nwin, piwin;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	if(numprocs > 8)
	{
		if(myid == 0)
			printf("Number of processors should be less than or equal to 8 \n");
		MPI_Finalize();
		exit(-1);
	}
	
        /* ..........Setting upu windows ...........*/
	if (myid ==0)
        {
	   MPI_Win_create(&n, sizeof(int), 1, MPI_INFO_NULL, MPI_COMM_WORLD, &nwin);
	   MPI_Win_create(&pi, sizeof(double), 1, MPI_INFO_NULL, MPI_COMM_WORLD, &piwin);
       }
       else
       {
	MPI_Win_create(MPI_BOTTOM, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &nwin);
	MPI_Win_create(MPI_BOTTOM, 0, 1, MPI_INFO_NULL,  MPI_COMM_WORLD, &piwin);
        }
 

      while (1) 
      {
        if (myid == 0) 
        {
	 printf("Enter the number of intervals: (Give 0 to quit) ");
	 fflush(stdout);
	 scanf("%d", &n);
	 pi = 0.0;

        } 

       MPI_Win_fence(0, nwin);

       if (myid != 0)
	 MPI_Get(&n, 1, MPI_INT, 0, 0, 1, MPI_INT, nwin);
         MPI_Win_fence(0, nwin);

       if (n == 0)
	  break;
       else 
       {
	  h	= 1.0 / (double) n;
	  sum 	= 0.0;
	  for 	(i = myid + 1; i <= n; i += numprocs) 
          {
		x = h * ((double)i - 0.5);
		sum += (4.0 / (1.0 + x*x));

	 } 
	 mypi = h * sum;
	 MPI_Win_fence( 0, piwin);

	 MPI_Accumulate(&mypi, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE,  MPI_SUM, piwin);
	 MPI_Win_fence(0, piwin);
 
	 if(myid == 0)
	    printf("pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi-PI25DT));
      }
  }
 
  MPI_Win_free(&nwin);
  MPI_Win_free(&piwin);

  MPI_Finalize();
 
  return 0;
} 
