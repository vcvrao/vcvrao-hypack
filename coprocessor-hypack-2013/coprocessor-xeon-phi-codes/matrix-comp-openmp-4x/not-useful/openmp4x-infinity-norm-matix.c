/* max=0;	
 *  for(i=0;i<Size;i++) {
 *  	sum = 0;
 *  		  for(j=0;j<Size;j++) {
 *  		     sum += (inarray[i][j]>=0)?(inarray[i][j]):(0-inarray[i][j]); 
 *  		          }
 *  		            max = max<sum ? sum : max;
 *  		             }
*/
#include <stdio.h>

 #define SIZE 4
 #pragma omp declare target

int reduce(int (*inarray)[SIZE])
{

  int sum = 0,max=0;
  #pragma omp target map(inarray[0:SIZE][0:SIZE]) map(sum) map(max)
  {
    for(int i=0;i<SIZE;i++)
    {
	    sum=0;
    	for(int j=0;j<SIZE;j++)
	{
    sum += (inarray[i][j]>=0)?(inarray[i][j]):(0-inarray[i][j]);
	}
	 max = max<sum ? sum : max;
  }
  }
  return sum;
}

int main()
{
  int inarray[SIZE][SIZE], sum, validSum,max=0;

  validSum=0;
  for(int i=0; i<SIZE; i++){
  	for(int j=0; j<SIZE; j++){
  inarray[i][j]=i+j+1;
  //validSum=i+1;
  }
  }
    for(int i=0;i<SIZE;i++)
    {
	    sum=0;
    	for(int j=0;j<SIZE;j++)
	{
    sum += (inarray[i][j]>=0)?(inarray[i][j]):(0-inarray[i][j]);
	}
	 max = max<sum ? sum : max;
  }
validSum=max;
 sum=0;
 sum = reduce(inarray);

 printf("sum reduction = %d,validSum=%d\n",sum, validSum);
}
