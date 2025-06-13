c
c****************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c    Example 2.2:	 : scatter.f
c
c    Objective           : To Scatter an integer array of size "n by 1"  
c                          using MPI Collective communication library call 
c                          (MPI_Scatter)
c
c                          This example demonstrates the use of
c                          MPI_Init
c                          MPI_Comm_rank
c                          MPI_Comm_size
c                          MPI_Bcast
c                          MPI_Scatter
c                          MPI_Finalize
c
c    Input               : Input Data file "sdata.inp".
c
c    Output              : Print the scattered array on all processes.
c
c    Necessary Condition : Number of processes should be 
c                          less than or equal to 8.
c
c   Created              : August-2013
c
c   E-mail               : hpcfte@cdac.in     
c
c*******************************************************************
c
c
	   program main

	   include "mpif.h"

           integer ScatterDataSize
	   parameter	(ScatterDataSize = 24 )

	   integer DataSize, Numprocs, MyRank, Root
  	   integer scatter_size

   	   integer InputBuffer(ScatterDataSize)
   	   integer RecvBuffer(ScatterDataSize)

C .........MPI Initialisation .......
	   call MPI_INIT( ierror )
  	   call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank,  ierror)
  	   call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs,ierror)

  	   Root = 0
	   if( MyRank .eq. Root) then
    	     open (unit= 18, FILE = "./data/sdata.inp")
    	     read(18,*) DataSize
             read(18,*)(InputBuffer(i), i=1,DataSize)     

    	     do i=1,DataSize,1   
		 print*,MyRank, i, InputBuffer(i)
    	     enddo 

	   endif

c 0123456789012345678901234567890123456789012345678901234567890123456789

         call MPI_Bcast(DataSize,1,MPI_INTEGER,Root,
     $                  MPI_COMM_WORLD,ierror)

	   if( mod( DataSize, Numprocs) .ne. 0) then
	    if(MyRank .eq. Root) write(*,*) 
     $	       "Input is not evenly divisible by Number of Processes"
	    goto 100
	   endif

C ......Scatter the Input Data to all processes ......
         
   	   scatter_size = DataSize/Numprocs
  	   call MPI_SCATTER (InputBuffer, scatter_size, MPI_INTEGER, 
     $                       RecvBuffer, scatter_size, MPI_INTEGER, 
     $                       Root, MPI_COMM_WORLD, ierror) 

	 do index = 1,scatter_size
	   print*,"MyRank = ", MyRank, " RecvBuffer(", index,")=", 
     $               RecvBuffer(index)
         enddo

 100	call MPI_FINALIZE( ierror )

	stop
	end





