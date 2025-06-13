c
c******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c    Example 2.7	 : Allreduce.f 
c
c    Objective           : To broadcast an integer array of size "n" by process with rank "0"  
c                          and perform Global summation using MPI_Allreduce
c                          (Combine values from all processes and distributed the result
c                           back to all processes)
c                        
c                          This example demonstrates the use of
c                          MPI_Init
c                          MPI_Comm_rank
c                          MPI_Comm_size
c                          MPI_Bast
c                          MPI_Allreduce                      
c                          MPI_Finalize
c
c    Input               : Input Data file "sdata.inp" by proces with Rank 0
c
c    Output              : Print the final sum on all processes.
c
c    Necessary Condition : Number of processes should be 
c                          less than or equal to 8.
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c*******************************************************************

	   program main

	   include "mpif.h"

         integer SendSumValue, RecvSumValue
	 integer Numprocs, MyRank, Root
         integer count, ierror

C .........MPI Initialisation .......
	 call MPI_INIT( ierror )
  	 call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank,  ierror)
  	 call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs,ierror)

  	 Root = 0
         if( MyRank .eq. Root) then
    	     open (unit= 18, FILE = "./data/SendSumValue.inp")
    	       read(18,*) SendSumValue
    	        print*, MyRank, SendSumValue
	   endif
c 
        call MPI_Bcast(SendSumValue,1,MPI_INTEGER,Root,
     $                  MPI_COMM_WORLD,ierror)

c .....Allreduce operation ....
         
        count = 1
        call MPI_Allreduce(SendSumValue, RecvSumValue, count, 
     $ MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierror) 

       print*,"MyRank = ", MyRank, "Sum =", RecvSumValue
 	 
      call MPI_FINALIZE(ierror)

	stop
	end





