c
c********************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c   Example 1.7		 : pie_pt_to_pt.f 
c
c    Objective           : To compute the value of PI by numerical integration.
c                          MPI Point-to-Point Communication library is used. 
c
c                          This example demonstrates the use of
c                          MPI_Init
c                          MPI_Comm_rank
c                          MPI_Comm_size
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
c   Created              : August-2013
c
c   E-mail               : hpcfte@cdac.in     
c
c*********************************************************************
      
      program main

      include 'mpif.h'

      double precision  PI25DT
      parameter        (PI25DT = 3.141592653589793238462643d0)

      double precision  mypi, pi, h, sum, x, f, a
      integer NoIntervals, MyRank, Numprocs, interval, Root
		integer Destination, Destination_tag, Source, Source_tag
		integer status(MPI_STATUS_SIZE)

C     function to integrate

      f(a) = 4.d0 / (1.d0 + a*a)

      call MPI_INIT( ierr )
      call MPI_COMM_RANK( MPI_COMM_WORLD, MyRank, ierr )
      call MPI_COMM_SIZE( MPI_COMM_WORLD, Numprocs, ierr )

      Root = 0
 10   if ( MyRank .eq. Root ) then
         write(6,98)
 98      format('Enter the number of intervals: (0 quits)')
         read(5,99) NoIntervals
 99      format(i10)

		 do 40 iproc = 1, Numprocs-1
			  Destination     = iproc
			  Destination_tag = 0
 	        call MPI_Send(NoIntervals, 1, MPI_INTEGER, Destination, 
     $			  Destination_tag, MPI_COMM_WORLD, ierror)
 40	 continue
		else
		    Source     = Root
		    Source_tag = 0
          call MPI_Recv(NoIntervals, 1, MPI_INTEGER, Source, Source_tag, 
     $		 MPI_COMM_WORLD, status, ierror)
      endif

c     check for quit signal
      if ( NoIntervals .LE. 0 ) then
        if(MyRank .eq. Root) print*,"Invalid Number of Intervals"
	goto 30
      endif

C     calculate the interval size

      h = 1.0d0/NoIntervals
      sum  = 0.0d0

      do 20 interval = MyRank+1, NoIntervals, Numprocs
         x = h * (dble(interval) - 0.5d0)
         sum = sum + f(x)
 20   continue
      mypi = h * sum

C     collect all the partial sums

      pi = 0.0
      if(MyRank .eq. Root) then
          pi = pi + mypi
	  do 50 iproc = 1, Numprocs-1
	      Source     = iproc
	      Source_tag = 0
              call MPI_Recv(mypi, 1, MPI_doUBLE_PRECISION, Source, 
     $			Source_tag, MPI_COMM_WORLD, status, ierror)       
              pi = pi + mypi
 50	   CONTINUE 
      else
	  Destination     = Root
	  Destination_tag = 0
          call MPI_Send(mypi, 1, MPI_doUBLE_PRECISION, Destination, 
     $			Destination_tag, MPI_COMM_WORLD, ierror)

      endif

C     node 0 prints the answer.
      if (MyRank .eq. Root) then
         write(6, 97) pi, abs(pi - PI25DT)
 97      format('  pi is approximately: ', F18.16,
     +          '  Error is: ', F18.16)
      endif

      goto 10

 30   call MPI_FINALIZE(ierr)
      stop
      end




