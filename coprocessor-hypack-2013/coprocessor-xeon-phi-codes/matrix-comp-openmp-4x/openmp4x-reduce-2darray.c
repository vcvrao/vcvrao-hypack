/***************************************************************************************************
* FILE		: openmp4x-reduce-2Darray.c
*
* INPUT		: Nil
*
* OUTPUT	: Displays Host and device output reduced sum
*
* CREATED	: August,2013
*
* EMAIL		: hpcfte@cdac.in
*
***************************************************************************************************/


#include <stdio.h>

 #define SIZE 10
 #pragma omp declare target

int reduce(int (*inarray)[SIZE])
{

  int sum = 0;
  #pragma omp target map(inarray[0:SIZE][0:SIZE]) map(sum)
  {
    for(int i=0;i<SIZE;i++)
    	for(int j=0;j<SIZE;j++)
    sum += inarray[i][j];
  }
  return sum;
}

int main()
{
  int inarray[SIZE][SIZE], sum, validSum;

  validSum=0;
  for(int i=0; i<SIZE; i++){
  	for(int j=0; j<SIZE; j++){
  inarray[i][j]=i+j;
  validSum+=i+j;
  }
  }

 sum=0;
 sum = reduce(inarray);

 printf("sum reduction = %d,validSum=%d\n",sum, validSum);
}
