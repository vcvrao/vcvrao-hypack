//Dot product of two vectors

#include <stdio.h>

 #define SIZE 4
 #pragma omp declare target

int reduce(int *Vector_1,int *Vector_2)
{

  int sum = 0;
  #pragma omp target map(Vector_1[0:SIZE]) map(Vector_2[0:SIZE]) map(sum)
  {
    for(int i=0;i<SIZE;i++)
    sum += Vector_1[i]*Vector_2[i];
  }
  return sum;
}

int main()
{
  int Vector_1[SIZE],Vector_2[SIZE], sum, validSum;

  validSum=0;
  for(int i=0; i<SIZE; i++){
  Vector_1[i]=i;
  Vector_2[i]=i;
  validSum+=Vector_1[i]*Vector_2[i];
  }

 sum=0;
 sum = reduce(Vector_1,Vector_2);

 printf("sum reduction = %d,validSum=%d\n",sum, validSum);
}
