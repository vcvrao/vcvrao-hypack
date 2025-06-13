/*
*************************************************************

	C-DAC Tech Workshop : hyPACK-2013
              October 15-18, 2013

   Example 1.2  : mpicpp-hello-world-slave.C
	
   Objective	: Write a MPI Program to print Heelo World
	
   Input	: None
   
   Output	: Message and Rank of the process.
    
   Created      : August-2013

   E-mail       : hpcfte@cdac.in     

*****************************************************************
*/
   
 
   #include<iostream>
   #include<stdio.h>
   #include<unistd.h>
   #include"mpi.h"
   
   using namespace std;
   
   int main(int argc,char *argv[])
   {
     int root=0,myrank,numprocs;
 
     MPI::Init(argc,argv);
     
     numprocs=MPI::COMM_WORLD.Get_size();
     myrank=MPI::COMM_WORLD.Get_rank();
      
     cout<<"\n Hello World From ::> " <<myrank<<"\n";  
     
     MPI::Finalize();
     return 0;
  }
 
