c
c*****************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c    Example 10:	 : gatherv.f 
c
c    Objective           : To gather non-uniform data from all the processes
c                          Gather data into specified locations from all 
c                          tasks in a Group
c
c                          This example demonstrates the use of
c                          MPI_Init
c                          MPI_Comm_rank
c                          MPI_Comm_size
c                          MPI_gather
c                          MPI_Barrier
c                          MPI_Finalize
c
c    Input               : Data from files (gdata0,gdata1,gdata2,gdata3,gdata4,
c                          gdata5,gdata6,gdata7) is available on respective 
c                          processes.
c
c    Output              : Gathered data is printed on each process.
c
c    Necessary Condition : Number of Processes should be equal to 8.
c
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c******************************************************************
c
	program main
	include "mpif.h"

C  	.......Variables Initialisation ......
	parameter (MAX_SIZE = 1000) 
        parameter (PROC_SIZE = 16)

	integer n_size, Numprocs, MyRank
	integer flag, total_count,Root, DataSize
	integer SendBuffer(MAX_SIZE), Output_A(MAX_SIZE)
     	integer  DisplacementVector(PROC_SIZE)
     	integer  RecvCount, RecvCountVector(PROC_SIZE)
        integer SendCount

	 integer  Destination, Destination_tag
	 integer  Source, Source_tag


C	 ........MPI Initialisation .......
	 call MPI_INIT(ierror) 
	 call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank,ierror)
	 call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs,ierror)

C        .......Read the Input file ......
          Root = 0
	  if(Numprocs .gt. PROC_SIZE) then
	     if(MyRank .eq. Root) then
                print*,"Number of processes  should be equql to 8"
                print*," Program Aborted  : Too many processes "
	     endif
	      goto 100
	  endif

	  if(MyRank .eq. 0) then
    	      open (unit= 15, FILE = "./gatherv-data/gdata0")
	  endif

	  if(MyRank .eq. 1) then
    	      open (unit= 16, FILE = "./gatherv-data/gdata1")
	  endif

	  if(MyRank .eq. 2) then
    	      open (unit= 17, FILE = "./gatherv-data/gdata2")
	  endif

	  if(MyRank .eq. 3) then
    	      open (unit= 18, FILE = "./gatherv-data/gdata3")
	  endif

	  if(MyRank .eq. 4) then
    	      open (unit= 19, FILE = "./gatherv-data/gdata4")
	  endif

	  if(MyRank .eq. 5) then
    	      open (unit= 20, FILE = "./gatherv-data/gdata5")
	  endif

	  if(MyRank .eq. 6) then
    	      open (unit= 21, FILE = "./gatherv-data/gdata6")
	  endif

	  if(MyRank .eq. 7) then
    	      open (unit= 22, FILE = "./gatherv-data/gdata7")
	  endif

	  if(MyRank .eq. 0) then
    	     read(15,*) n_size
             read(15,*) (SendBuffer(i), i=1,n_size)     
    	     write(6,201) MyRank, n_size,(SendBuffer(i), i= 1, n_size) 
	  endif

	  if(MyRank .eq. 1) then
    	     read(16,*) n_size
             read(16,*) (SendBuffer(i), i=1,n_size)     
    	     write(6,201) MyRank, n_size,(SendBuffer(i), i= 1, n_size) 
	  endif

	  if(MyRank .eq. 2) then
    	     read(17,*) n_size
             read(17,*) (SendBuffer(i), i=1, n_size)     
    	     write(6,201) MyRank, n_size,(SendBuffer(i), i= 1, n_size) 
	  endif

	  if(MyRank .eq. 3) then
    	     read(18,*) n_size
             read(18,*) (SendBuffer(i), i=1,n_size)     
    	     write(6,201) MyRank, n_size,(SendBuffer(i), i= 1, n_size) 
	  endif

	  if(MyRank .eq. 4) then
    	     read(19,*) n_size
             read(19,*) (SendBuffer(i), i=1,n_size)     
    	     write(6,201) MyRank, n_size,(SendBuffer(i), i= 1, n_size) 
	  endif

	  if(MyRank .eq. 5) then
    	     read(20,*) n_size
             read(20,*) (SendBuffer(i), i=1,n_size)     
    	     write(6,201) MyRank, n_size,(SendBuffer(i), i= 1, n_size) 
	  endif

	  if(MyRank .eq. 6) then
    	     read(21,*) n_size
             read(21,*) (SendBuffer(i), i=1,n_size)     
    	     write(6,201) MyRank, n_size,(SendBuffer(i), i= 1, n_size) 
	  endif

	  if(MyRank .eq. 7) then
    	     read(22,*) n_size
             read(22,*) (SendBuffer(i), i=1,n_size)     
    	     write(6,201) MyRank, n_size,(SendBuffer(i), i= 1, n_size) 
	  endif

	  if(MyRank .eq. 8) then
    	     read(23,*) n_size
             read(23,*) (SendBuffer(i), i=1,n_size)     
    	     write(6,201) MyRank, n_size,(SendBuffer(i), i= 1, n_size) 
	  endif
c
         total_count = 0

C        .........The REDUCE function of MPI : Determine total array size.
         call MPI_REDUCE(n_size,total_count,1,MPI_INTEGER,MPI_SUM, 
     $          Root, MPI_COMM_WORLD, ierror)
 
c        .........Array Bounds Check & abort program check...........
         flag = 1
         if(MyRank .eq. Root) then
c           print *, "GatherV Array size ", total_count
            if(total_count .gt. MAX_SIZE) flag = 0
         endif

         call MPI_Bcast(flag,1,MPI_INTEGER,Root,MPI_COMM_WORLD,ierror)

         if(flag .eq. 0) then
           print *, " Upper bound of a Gather Array exceeds : Aborted "
           go to 100
         endif
c
c        ...............Gather size of array on each process ..........
c        ...............Send RecvCount to each process ........ 
	 Destination_tag = 0
	 Source_tag = 0
	 if(MyRank .eq. Root) then
             RecvCountVector(1) = n_size 
c 	     write(6,*) "Recv", MyRank, RecvCountVector(1)
	     do iproc = 1, Numprocs-1	   
             Source = iproc
	     DataSize = 1

             call MPI_RECV(RecvCount, DataSize, MPI_INTEGER, Source, 
     $       Source_tag, MPI_COMM_WORLD, status, ierror)

             RecvCountVector(iproc+1) = RecvCount 
c 	     write(6,*) "Recv", iproc+1, MyRank, RecvCountVector(iproc+1)
           end do
       else
          Destination = 0
          Datasize = 1
	    call MPI_SEND(n_size,DataSize,MPI_INTEGER,Destination, 
     $	   Destination_tag, MPI_COMM_WORLD, ierror)
        endif
c
c        .........Broadcast the RecvCountVector
         call MPI_Bcast(RecvCountVector,Numprocs,MPI_INTEGER,Root,
     $                MPI_COMM_WORLD,ierror)
c
c	write(6,*) "Recv - MyRank ", MyRank 
c 	write(6,200) (RecvCountVector(i), i= 1, Numprocs) 

      ivalue = 2
      if(ivalue .eq. 1) then
       print *, " ****** Terminated  SEND-RECV ", MyRank
       goto 100
      endif

C     .............Initial arrangement for Gatherv operation ....
C     .....Determine "Displacement Vector "
      if (MyRank .eq. Root) then
c
		DisplacementVector(1) = 0
		do iproc = 2, Numprocs
	   	  disp_value = 0
	      	  do jproc = 1, iproc-1
	           disp_value = disp_value + RecvCountVector(jproc)
		  enddo
		  DisplacementVector(iproc) = disp_value 
		enddo
      endif
C
       call MPI_Bcast(DisplacementVector, Numprocs,MPI_INTEGER,Root,
     $                MPI_COMM_WORLD,ierror)

c	write(6,*) "DisplacementVector - MyRank ", MyRank 
c 	write(6,200) (DisplacementVector(i), i= 1, Numprocs) 

       ivalue = 2
       if(ivalue .eq. 1) then
        print *, " ****** Terminated  SEND-RECV ", MyRank
        goto 100
       endif
C   ......... Gather Vector A .........
c
      SendCount = n_size 
      call MPI_Gatherv(SendBuffer, SendCount, MPI_INTEGER, 
     $     Output_A, RecvCountVector,DisplacementVector,
     $     MPI_INTEGER, Root, MPI_COMM_WORLD,ierror)

      if(MyRank .eq. root) then
    	  write(6,*) " "
    	  write(6,*) "  Final Results : Gathering Data on Process ", MyRank
    	  write(6,*) "  Total Gatheres Entries ", total_count
	  write(6, 200) (Output_A(i), i = 1,total_count)
       endif

200    format(10i3)
201    format(2x,"MyRank =", i4, 3x, "n_size =",i4,4x,"Entries =",10i3)
100    call MPI_FINALIZE(ierror) 
   	  stop
   	  end




