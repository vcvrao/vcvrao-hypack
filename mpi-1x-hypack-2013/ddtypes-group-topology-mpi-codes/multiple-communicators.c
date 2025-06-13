/*
****************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

   Example 4.6	     : multiple-communicators.c

   Objective         : To create a diagonal communicator group of processes
                       in a square grid processes on cluster

   Input             : None  

   Output            : Print the list of Process in the diagonal 
                       communicator on each process

   Created           : August-2013

   E-mail            : hpcfte@cdac.in     

****************************************************************** */

  #include<stdio.h>
  #include"mpi.h"

  int main(int argc,char **argv)
  {
 
    int Numprocs,Myrank;
    MPI_Group group_world;

/*.....Initialize MPI...........................*/
    
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&Numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&Myrank);

/*...Initialize variable for new communicator group...........*/
    int   even_id,iproc,Root=0,i=0;
    int   *even_member,Neven;
    char  str[12],msg[12];
    MPI_Group  even_group;
    MPI_Comm   even_comm;
    MPI_Status status;


/*..make the list of processes in the new Communicator..*/ 
    
   Neven = (Numprocs+1)/2;
   even_member=(int *)malloc(Neven *sizeof(int));
   for(i=0 ; i < Neven ; i++)
      even_member[i]=2*i;

/*....Get the underlying MPI_COMM_WORLD...........*/
    MPI_Comm_group(MPI_COMM_WORLD,&group_world);

/*.. Create the new group...*/ 
   MPI_Group_incl(group_world,Neven,even_member,&even_group);

/*...Create the new communicator............*/
   MPI_Comm_create(MPI_COMM_WORLD,even_group,&even_comm);

/*..Get the rank of processes in new communicator group...*/
   MPI_Group_rank(even_group,&even_id);

/*..Send the message from new communicator process 0 to other process....*/
if(even_id != MPI_UNDEFINED)
{
   if( even_id == 0)
   {
     strcpy(str,"HELLO_WORLD");
     for(iproc=1;iproc<Neven;iproc++)
     {
       MPI_Send(str,12,MPI_CHAR,iproc,0,even_comm);
     }

   }

/*.....Receive the message & print it........*/ 
   if(even_id != 0)
   {
     MPI_Recv(msg,12,MPI_CHAR,0,0,even_comm,&status);
     printf("\nOld Rank is :%d New Rank is :%d get %s from process %d\n",Myrank,even_id,msg,Root); 


   }
}
/*MPI Finalizing*/
MPI_Finalize();
} 
