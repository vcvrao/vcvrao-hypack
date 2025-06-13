c
c*******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c                         C-DAC Tech Workshop : HeGaPa-2012
c                            July 16-20,2012
c
c******************************************************************
c
c    Example 2.1         : broadcast.f
c
c    Objective           : To braodcast an integer array of size "n" by 
c                           "with Rank 0"  using MPI Collective 
c                          communication library call 
c                          (MPI_Bcast)
C
c                          This example demonstrates the use of
c                          MPI_Init
c                          MPI_Comm_rank
c                          MPI_Comm_size
c                          MPI_Bcast                      
c                          MPI_Finalize
c
c    Input               : Input Data file "sdata.inp" by proces with Rank 0
c
c    Output              : Print the scattered array on all processes.
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

	   parameter (MaxSize = 100 )

           integer broadcastDataSize
	   integer Numprocs, MyRank, Root
   	   integer InputBuffer(MaxSize)

C .........MPI Initialisation .......
	   call MPI_INIT( ierror )
  	   call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank,  ierror)
  	   call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs,ierror)

  	   Root = 0
	   if( MyRank .eq. Root) then
    	     open (unit= 18, FILE = "./data/broadcast-data.inp")
    	       read(18,*) broadcastDataSize
               read(18,*)(InputBuffer(i), i=1,broadcastDataSize)  
   
             write(6, 200) MyRank
             write(6, 201) (InputBuffer(j), j = 1, broadcastDataSize)

	   endif
c 
	if(Numprocs .ne. 4) then
         write(6,*) 'Input data file works for 4 processes '
         write(6,*) 'Change the data file large number of processes'
         write(6,*) 
         write(6,*) 'Program Aborted :Input No. of Process is wrong'
         write(6,*) 'Program is written for only 4 processes. !!'
         write(6,*) 
c         go to 100
  	endif 
 
c
        call MPI_Bcast(Numprocs,1,MPI_INTEGER,Root,
     $                  MPI_COMM_WORLD,ierror)

	If(Numprocs .ne. 4) go to 100      
        call MPI_Bcast(broadcastDataSize,1,MPI_INTEGER,Root,
     $                  MPI_COMM_WORLD,ierror)

c .....Broadcast the Input Data to all processes ......
          
  	  call MPI_Bcast(InputBuffer, broadcastDataSize, MPI_INTEGER,                
     $                       Root, MPI_COMM_WORLD, ierror) 

       if(MyRank .ne. 0) then
          write(6, 200) MyRank
          write(6, 201) (InputBuffer(j), j = 1, broadcastDataSize)
200	format( 5x, " Rank of Process ", I4)
201	format(24I3)
       endif

100 	call MPI_FINALIZE(ierror)

	stop
	end





