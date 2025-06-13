c
c********************************************************************
c
c	C-DAC Tech Workshop : hyPACK-2013
c               October 15-18, 2013
c
c   Example 11		: pie_collective.f
c
c   Objective           : To compute the value of PI by numerical integration.
c                         MPI Collective communication library calls are used. 
c                         This example demonstrates the use of
c
c                         MPI_Init
c                         MPI_Comm_rank
c                         MPI_Comm_size
c                         MPI_Bcast
c                         MPI_Reduce
c                         MPI_Finalize
c
c   Input               : The number of intervals.
c
c   Output              : The calculated value of PI.
c
c   Necessary Condition : Number of Processes should be 
c                         less than or equal to 8.
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

      double precision  mypi, pi, h, sum, x, f, a
      integer NoInterval, MyRank, Numprocs, interval, Root

      
C     function to integrate

      f(a) = 4.0d0 / (1.0d0 + a*a)

      call MPI_INIT( ierr )
      call MPI_COMM_RANK( MPI_COMM_WORLD, MyRank, ierr )
      call MPI_COMM_SIZE( MPI_COMM_WORLD, Numprocs, ierr )

      Root = 0 
     
 10   if ( MyRank .eq. Root ) then
         write(6,98)
 98      format('Enter the number of intervals: (0 quits)')
         read(5,99) NoInterval
 99      format(i10)
      endif
      
      call MPI_BCAST(NoInterval,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)

c     check for quit signal
      if ( NoInterval .LE. 0 ) then
        if(MyRank .eq. Root) print*,"Invalid Number of Intervals"
	goto 30
      endif

C     calculate the interval size

      h = 1.0d0/NoInterval
      sum  = 0.0d0

      do 20 interval = MyRank+1, NoInterval, Numprocs
         x = h * (dble(interval) - 0.5d0)
         sum = sum + f(x)
 20   continue
      mypi = h * sum

C     collect all the partial sums
      call MPI_REDUCE(mypi,pi,1,MPI_DOUBLE_PRECISION,MPI_SUM,Root,
     $     MPI_COMM_WORLD,ierr)

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




