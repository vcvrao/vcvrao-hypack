
/*
*********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

   Example 4.1 : general-derived-datatype.c

   Objective   : To build a general derived datatypes in which process 
                 with rank 0 broadcast	struct having two float and 
                 an int values

   Input       : None

   Output      : Each Process prints the elements of struct 

   Created     : August-2013

   E-mail      : hpcfte@cdac.in     


**********************************************************************
*/


#include<stdio.h>
#include"mpi.h"
int main(int argc,char **argv)
{
  int Group_size,Rank,root=0,tag=0,i,iproc;
  MPI_Status status;
  
/*.......Creating  structure..................................*/
  struct INDATA 
  {
   int   a;
   float b[4];
   char  c[4];
  };
 
/*.....Creating object of structure.........................*/
  struct INDATA indata,outdata;
 
/*..Build Datatype describing the the structure components.....*/ 
  
  MPI_Datatype new_type;
  int block_lengths[3];
  MPI_Aint displacement[3];
  MPI_Datatype typelist[3];
  MPI_Aint address,start_address;
  int base;
 
/*.........MPI initialization...............................*/                   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
   MPI_Comm_size(MPI_COMM_WORLD,&Group_size);


/*..........Read the input...................................*/
  if(Rank==0)
  {
    printf("\nEnter the Integer value for a :\n");
    fflush(stdout);
     scanf("%d",&indata.a);
  
    printf("\nEnter the Float values for array b :\n");
    for(i=0;i<4;i++)
      scanf(" %f ",&indata.b[i]);

     printf("\nEnter the Character values for array c :\n");
     for(i=0;i<4;i++)
       scanf(" %c ",&indata.c[i]);
  }                                                               
 

/*........Assign the No. of element of each structure component..............*/

   block_lengths[0]=1;
   block_lengths[1]=4;
   block_lengths[2]=4;
    
/*........Assign Type of each structure component...................*/

  typelist[0]=MPI_INT;
  typelist[1]=MPI_FLOAT;
  typelist[2]=MPI_CHAR;

/*......Compute displacements of structure components.....................*/
  displacement[0]=0;
 
  MPI_Address(&indata.a , &start_address);
  MPI_Address(&indata.b , &address);
  displacement[1]=address-start_address;

  MPI_Address(&indata.c , &address);
  displacement[2]=address-start_address;
  
/*...Build derived struct type & commit it ..............................*/
  MPI_Type_struct(3,block_lengths,displacement,typelist,&new_type);
  MPI_Type_commit(&new_type);

/*.....Call MPI Broadcast...................................*/
  MPI_Bcast(&indata,1,new_type,0,MPI_COMM_WORLD); 

 
/*...........Send the entire structure from Root Process......................*/
  if(Rank==0)
  {
    for( iproc=1 ; iproc < Group_size ;iproc++)
        MPI_Send(&indata,1,new_type,iproc,tag,MPI_COMM_WORLD);
  }

/*...Receiving the structure & Display the Result........................*/
  if(Rank != 0) 
  {
 
      MPI_Recv(&outdata,1,new_type,0,tag,MPI_COMM_WORLD,&status);

      printf("\tReceived Data on %d  process from process %d",Rank,status.MPI_SOURCE);  
      printf("\n-----------------------------------------------------\n");
      printf("\n a=%d\n ",outdata.a); 
  
      for(i=0;i<4;i++)
        printf("\n b[%d]=%f \t c[%d]=%c\n ",i,outdata.b[i],i,outdata.c[i]);
    }  


/*.....MPI finalizing................................*/
 MPI_Finalize();
}
  
