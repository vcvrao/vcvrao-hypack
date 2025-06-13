/***********************************************************************************************************
* FILE		: openmp4-0-distribute-1Darray-sum.c
*
* INPUT		: #Vector size #Team size #Number of threads per team
*
* OUTPUT	: sum of elements of a given 1D array
*
* Description	: In this code elements of a given 1D array is distributed among teams. Each team computes
* 		  partial reduced sum.At last Master thread implicitl gather partial sums from each team 
* 		  and computes final sum.
*
* CREATED	: August ,2013
*
* EMAIL		: hpcfte@cdac.in
*
* ********************************************************************************************************/

#include <stdio.h>
#include<stdlib.h>
#define SIZE 1000
#define TEAM_SIZE 2
#define NUM_THREADS 100

int main()
{
  int A[SIZE], sum=0;

  int i,j;
  if(SIZE%TEAM_SIZE!=0)
  {
	  printf("Size must be divisible by team size\n");
	  exit(1);
  }

   for (i =0; i < SIZE; i++)
   A[i] = i; 

#pragma omp target map(to : A[0:SIZE])
#pragma omp teams num_teams(TEAM_SIZE) num_threads(NUM_THREADS) 
#pragma omp distribute
   for(i = 0; i < SIZE; i +=( SIZE/TEAM_SIZE))
#pragma omp parallel for reduction(+:sum)
   for (j = i; j < i + (SIZE/TEAM_SIZE); j++)
   sum += A[j]; 

 printf("sum = %d\n",sum);
}
