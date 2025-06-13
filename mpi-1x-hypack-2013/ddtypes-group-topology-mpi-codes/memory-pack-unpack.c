
/*
*******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

   Example 4.5	     : memory-pack-unpack.c

   Objective         : Pack the different datatype variable on the process 0 
                       & send to other processes & unpack the data on that 
                       processes & displayresults

   Input             : The Process with rank 0 reads the three values a, b,   
                       and n. format. 

   Output            : Process with Rank greater than 1 prints the output 
                       values a, b, and n 

   Created           : August-2013

   E-mail            : hpcfte@cdac.in     

******************************************************************
*/


#include<stdio.h>
#include<stdlib.h>
#include"mpi.h"
#define n_size 6

int main(int argc , char **argv)
{
    int Myrank, Numprocs,iproc;
    int array[n_size],index=0;
    float b=1.2;
    char c='c', buffer[100]; 
    int position;
    MPI_Status status;
 
/*.....MPI initialization....................................*/
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&Myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&Numprocs);


/*....Read the Input.............................*/
  if(Myrank==0)
  {
        for(index=0 ; index<n_size ;index++)
         {
           array[index]=index;
         } 

        printf("-------------------------------------------------------\n"); 
        printf(" PACKED DATA IS :\n");
        printf("--------------------------------------------------------\n"); 


/*....Print the Data to be packed......................*/

        printf("1.Array of integers :\n"); 
        for(index=0 ; index<n_size ; index++)
         {
            printf(" array[%d]=%d\t",index,array[index]);
         }
         printf(" \n2.Float Value  : %f ",b);
         printf("\n3.Character Value:%c\n",c);
 
/*...Packed the Data into buffer on process 0 & send the packed data......*/
        position=0;

        MPI_Pack(&array,n_size,MPI_INT,buffer,100,&position,MPI_COMM_WORLD);
        MPI_Pack(&b,1,MPI_FLOAT,buffer,100,&position,MPI_COMM_WORLD);
        MPI_Pack(&c,1,MPI_CHAR,buffer,100,&position,MPI_COMM_WORLD);
  
        for(iproc=1 ;iproc<Numprocs ;iproc++)  
           MPI_Send(buffer,position,MPI_PACKED,iproc,0,MPI_COMM_WORLD);
  }
     MPI_Barrier(MPI_COMM_WORLD);  

/*...Unpacked the data & print the output.....................*/
  if(Myrank != 0)
  {
      MPI_Recv(buffer,100,MPI_PACKED,0,0,MPI_COMM_WORLD,&status);
      
      position=0;
      MPI_Unpack(buffer,100,&position,&array,n_size,MPI_INT,MPI_COMM_WORLD);
      MPI_Unpack(buffer,100,&position,&b,1,MPI_FLOAT,MPI_COMM_WORLD);
      MPI_Unpack(buffer,100,&position,&c,1,MPI_CHAR,MPI_COMM_WORLD);
      
      printf("------------------------------------------------------\n"); 
      printf("UNPACKED DATA ON PROCESS %d :\n ",Myrank);
      printf("--------------------------------------------------------\n"); 
     
      printf("1.Array of integers :\n"); 
 
      for(index=0;index<n_size;index++)
        printf("array[%d]=%d\t",index,array[index]);

      printf("\n2.Unpacked Float value is :%f ",b);
      printf("\n3.Unpacked character value is :%c\n ",c);
  
   }  

/*....MPI Finalizing.....*/
 MPI_Finalize();
} 

