c
c********************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c                   Example 4 (Tools_Pi_Idleness.f)
c
c
c    Objective           : To compute the value of PI by numerical integration.
c                          MPI Point-to-Point Communication and collective
c                          communication libraries are used to show idleness in c program.
c
c                          This example demonstrates the use of
c                          MPI_Init
c                          MPI_Comm_rank
c                          MPI_Comm_size
c                          MPI_Bcast
c                          MPI_Send
c                          MPI_Recv
c                          MPI_Finalize
c
c    Input               : The number of intervals
c
c    Output              : The calculated value of PI.
c
c    Necessary Condition : Number of Processes should be
c                          less than or equal to 8.
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c
c*********************************************************************      
      program main

      include 'mpif.h'

      double precision  PI25DT
      parameter        (PI25DT = 3.141592653589793238462643d0)

      double precision  mypi, h, sum, x, f, a
      integer NoofIntervals, myid, numprocs, Root
		integer status(MPI_STATUS_SIZE)

C     function to integrate

      f(a) = 4.d0 / (1.d0 + a*a)

      call MPI_INIT( ierr )
      call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
      call MPI_COMM_SIZE( MPI_COMM_WORLD, numprocs, ierr )

      Root = 0
      if ( myid .eq. Root ) then
      NoofIntervals = 10 
      end if
      call MPI_BCAST(NoofIntervals,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
       if ( myid .eq. Root) then
          h = 1.0d0/NoofIntervals
          sum  = 0.0d0
          do 20 i = myid+1, NoofIntervals, numprocs
             x = h * (dble(i) - 0.5d0)
             sum = sum + f(x)
 20       continue
         mypi = h * sum
 	        call MPI_Send(mypi, 1, MPI_DOUBLE_PRECISION, myid+1, 
     $			  0, MPI_COMM_WORLD, ierror)
            endif
		 if( myid .lt. numprocs-1) then
          call MPI_Recv(mypi, 1, MPI_DOUBLE_PRECISION, myid-1,0, 
     $		 MPI_COMM_WORLD, status, ierror)
          h = 1.0d0/NoofIntervals
          sum  = 0.0d0
          do 40 i = myid+1, NoofIntervals, numprocs
             x = h * (dble(i) - 0.5d0)
             sum = sum + f(x)
 40       continue
         mypi = mypi +  h * sum
 	        call MPI_Send(mypi, 1, MPI_DOUBLE_PRECISION, myid+1, 
     $			  0, MPI_COMM_WORLD, ierror)
         endif
         if(myid .eq. numprocs-1) then
          call MPI_Recv(mypi, 1, MPI_DOUBLE_PRECISION, myid-1,0,
     $           MPI_COMM_WORLD, status, ierror)
          h = 1.0d0/NoofIntervals
          sum  = 0.0d0
          do 50 i = myid+1, NoofIntervals, numprocs
             x = h * (dble(i) - 0.5d0)
             sum = sum + f(x)
 50       continue
          mypi = mypi + h * sum 
         write(6, 97) mypi, abs(mypi - PI25DT)
 97      format('  pi is approximately: ', F18.16,
     +          '  Error is: ', F18.16)
          endif
       if ( NoofIntervals .eq. 0 ) then  
       goto 30
       end if
 30   call MPI_FINALIZE(ierr)
      stop
      end





