/*****************************************************************************
                   C-DAC : M.TECH(HPCS)
                   BATCH 2015-17 
                        

 Example 1             : omp-quicksort.c

 Objective             : Write an OpenMP program to Sort the Elements using                              QUICKSORT of One-Dimensional real array.
                         This example demonstrates the use of OpenMP
                         Parallel For Directive .
 
 Input                 : Size of an array
                         Number of threads 

 Output                : Sorted elements of an array                                               
                                                                        
 Created               : December 2015

 E-mail                : hpcfte@cdac.in     

************************************************************************/


 #include<stdio.h>
#include <omp.h>
#include<stdlib.h>
#include<sys/time.h>
void quicksort(int [10000],int,int);

int main(int argc ,char **argv){
  int x[10000],size,i,threadid,Noofthreads;
struct timeval  TimeValue_Start;
	struct timezone TimeZone_Start;

	struct timeval  TimeValue_Final;
	struct timezone TimeZone_Final;
	long            time_start, time_end;
        double          time_overhead;


if(argc != 3){
printf("\t\t very few Arguments\n ");
printf("\t\t Syntax: exec <Threads><array-size>\n");
}
Noofthreads = atoi(argv[1]);
if((Noofthreads !=1) && (Noofthreads !=2) &&(Noofthreads !=4) && (Noofthreads !=8) && (Noofthreads != 16)){
printf("\n Number of threads should be 1,2,4,8 or 16 for the execution of program. \n\n");
exit(-1);
}
size=atoi(argv[2]);
 // printf("\n\t\tEnter size of the array \n ");
  //scanf("%d",&size);
if(size <= 0)
{
printf("\n\t\t Array size should be of Positive Value\n ");
exit(-1);
}
printf("\n\t\t Threads : %d ",Noofthreads);
printf("\n\t\t Array size : %d",size);
if(size%Noofthreads != 0)
{
printf("\n\t\tArray should be divided in equal size\n");
exit(-1);
}
else
{ 
//printf("\n\t\t Enter the elements\n ");
for(i=0;i< size;i++)
{

//scanf("%d",&x[i]);
x[i]=rand()%10000;
}
gettimeofday(&TimeValue_Start, &TimeZone_Start);
omp_set_num_threads(Noofthreads);
#pragma omp parallel
{
#pragma omp critical
  quicksort(x,0,size-1);
}
gettimeofday(&TimeValue_Final, &TimeZone_Final);
time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
	time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
	time_overhead = (time_end - time_start)/1000000.0;

}
printf("\n\t\t Sorted elements:\n ");
for(i=0;i<size;i++){
   printf(" %d",x[i]);
printf("\n");
//printf("\n\t\t Time in Seconds (T)        : %lf Seconds \n",time_overhead);
}
printf("\n\t\t Time in Seconds (T)        : %lf Seconds \n",time_overhead);

  return 0;
}

void quicksort(int x[10],int first,int last)
{
int pivot,j,temp,i;

if(first<last)
{
pivot=first;
i=first;
j=last;
while(i<j)
{
while(x[i]<=x[pivot]&&i<last)
i++;
while(x[j]>x[pivot])
j--;
if(i<j)
{
temp=x[i];
x[i]=x[j];
x[j]=temp;
 }
 }
temp=x[pivot];
x[pivot]=x[j];
x[j]=temp;
quicksort(x,first,j-1);
quicksort(x,j+1,last);

}
}



