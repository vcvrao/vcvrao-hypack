c
c*******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c*****************************************************************
c
c  Example 1.1   		: hello_world.f
c
c  Objective           : MPI Program to print "Hello World"
c
c                        This example demonstrates the use of
c                        MPI_Init
c                        MPI_Comm_rank
c                        MPI_Comm_size
c                        MPI_Send
c                        MPI_Recv
c                        MPI_Finalize
c
c  Input               : Message="Hello World"
c
c  Output              : Message and Rank of the process. 
c
c  Necessary Condition : Number of Processes should be 
c                        less than or equal to 8.
c
c   Created           : August-2013
c
c   E-mail            : hpcfte@cdac.in     
c
c********************************************************************

	 program main

	 include "mpif.h"

	 integer  MyRank, Numprocs
   	 integer  Destination, Source, iproc
	 integer  Destination_tag, Source_tag 
	 integer  Root, MessageSize
	 integer  status(MPI_STATUS_SIZE)
         integer  ierror
	 character*12   SendMessage, RecvMessage 

	 data SendMessage/'Hello World'/

C.........MPI initialization.... 

    	 call MPI_INIT(ierror)
    	 call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs, ierror)
    	 call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank, ierror)

	 Root = 0
	 MessageSize = 11

        if(MyRank .ne. Root) then
	  Destination = Root
	  Destination_tag = 0
	  call MPI_SEND(SendMessage, MessageSize, MPI_CHARACTER, Destination, 
     $  	Destination_tag,  MPI_COMM_WORLD, ierror)
	  else 
	   do iproc = 1, Numprocs-1
	     Source = iproc
	     Source_tag = 0
	     call MPI_RECV(RecvMessage, MessageSize, MPI_CHARACTER, Source, 
     $	     Source_tag, MPI_COMM_WORLD, status, ierror)
	        print *, RecvMessage,' from Process with Rank ', iproc
	   enddo

	 endif

         call MPI_FINALIZE( ierror )

         stop
         end



