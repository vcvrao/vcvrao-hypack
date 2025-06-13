c
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c
c			  Example 29
c			 MAIN PROGRAM
c
c*******************************************************************
c  Objective : Find a solution to the Poisson problem using Jacobi 
c              iteration method on a 1-d decomposition
c
c  Input     : The size of the domain and number of iterations are read
c              by process 0 and broadcast to all other processes. 
c
c  Output    : The difference is printed out at each step.
c              The Jacobi iteration is run until the change in 
c              successive elements is small or a maximum number of 
c              iterations is reached.  
c 
c*******************************************************************
c
      program main 

      include "mpif.h"
      integer 	maxn

      parameter (maxn = 500)
      parameter (iter = 20000)
      parameter (tolerance  = 0.000001)
      real*8    a(maxn,maxn), b(maxn,maxn), f(maxn,maxn)

      integer 	nx, ny, myid, Root, numprocs,iter, ierr
      integer 	comm1d, nbrbottom, nbrtop, s, e, it

      real*8 	diff, diffnorm, dwork
      real*8 	t1, t2

      Root = 0
      CALL MPI_INIT( ierr )
      CALL MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
      CALL MPI_COMM_SIZE( MPI_COMM_WORLD, numprocs, ierr )
 
      if (myid .eq. Root) then
c       Get the number of cells in X- and Y- direction
        write(6,100)
100     format(//3x,'Number of Cells in X- and Y- direction is same'/
     $  5x,' Give number of cells in X-direction')
	read(5,*) nx
      endif

      CALL MPI_BCAST(nx,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)

      if( (mod(nx,numprocs) .ne. 0) .or. (nx .ge. maxn) ) then
       if(myid .eq. Root) write(6,103)
103     format(4x,'Number of cells exceeds the dimension of the array'/
     $   'Number of cells should be divisible by  no of processors')
        go to 30
      endif
      ny = nx

c     Get a new communicator for a decomposition of the domain

      CALL MPI_CART_CREATE( MPI_COMM_WORLD, 1, numprocs, .false., 
     $                    .true., comm1d, ierr )

c     Get my position in this communicator, and my neighbors
 
      CALL MPI_COMM_RANK( comm1d, myid, ierr )
      CALL MPI_Cart_shift( comm1d, 0,  1, nbrbottom, nbrtop, ierr   )

c     Compute the actual decomposition
     
      CALL MPE_DECOMP1D( ny, numprocs, myid, s, e )

c     Initialize the right-hand-side (f) and the initial solution guess (a)

      CALL onedinit( a, b, f, nx, s, e )

c     Actually do the computation. Note the use of a collective 
c     operation to check for convergence, and a do-loop to bound the 
c     number of iterations.

      CALL MPI_BARRIER( MPI_COMM_WORLD, ierr )

      t1 = MPI_WTIME()
 
      do 10 it=1, iter

	CALL exchng1( a, nx, s, e, comm1d, nbrbottom, nbrtop )
	CALL sweep1d( a, f, nx, s, e, b )
	CALL exchng1( b, nx, s, e, comm1d, nbrbottom, nbrtop )
	CALL sweep1d( b, f, nx, s, e, a )

	dwork = diff( a, b, nx, s, e )

	CALL MPI_Allreduce( dwork, diffnorm, 1, MPI_DOUBLE_PRECISION, 
     $                      MPI_SUM, comm1d, ierr )

        if (diffnorm .lt. tolerance) goto 20

        if (myid .eq. Root) print *, it, ' Difference is ', diffnorm
10    continue

      t2 = MPI_WTIME()

      if (myid .eq. Root) then 
          print *, 'Failed to converge after', it, ' Iterations in', 
     $ 	            t2 - t1, ' seconds '
        endif
	 GOTO 30
20    continue

      t2 = MPI_WTIME()
      if (myid .eq. Root) then
        write(6,105) it, t2 -t1
105     format(4x,'Converged after ',2x, i8,'  Iterations and the time
     $  taken is ', F20.10//
     $  15x, '*** Successful exit ***')
      endif
c
30    call MPI_FINALIZE(ierr)

      stop
      end
c
c     ----------------------------------------------------------
c
c     This subroutine initialize all the arrays used in the program and
c     impose boundary conditions, which are Dirichelt type.
c
      SUBROUTINE onedinit( a, b, f, nx, s, e )
c
      integer  nx, s, e
      real*8   a(0:nx+1,s-1:e+1),b(0:nx+1,s-1:e+1),f(0:nx+1,s-1:e+1)

c     Local variables
      integer i, j
c
      do 10 j=s-1,e+1
         do 10 i=0,nx+1
            a(i,j) = 0.0d0
            b(i,j) = 0.0d0
            f(i,j) = 0.0d0
 10      continue
c      
c    Handle boundary conditions
c
      do 20 j=s,e
         a(0,j)    = 1.0d0
         b(0,j)    = 1.0d0
         a(nx+1,j) = 0.0d0
         b(nx+1,j) = 0.0d0
20    continue

      if (s .eq. 1) then
         do 30 i=1,nx
            a(i,0) = 2.0d0
            b(i,0) = 2.0d0
30       continue 
      endif
c
      return
      end
c
c     ----------------------------------------------------------
C
C     This file contains a routine for producing a decomposition of 
C     a 1-d array into no of processors.  It may be used in "direct" 
C     product decomposition.  The values returned assume a "global" 
C     domain in [1:n]
C
      SUBROUTINE MPE_DECOMP1D( n, numprocs, myid, s, e )
C
C
      integer n, numprocs, myid, s, e, nlocal, deficit

      nlocal  = n / numprocs
      s	      = myid * nlocal + 1
      deficit = mod(n,numprocs)
      s	      = s + min(myid,deficit)

C     Give one more slice to processors

      if (myid .lt. deficit) then
          nlocal = nlocal + 1
      endif

      e = s + nlocal - 1

      if (e .gt. n .or. myid .eq. numprocs-1) e = n

      return
      end
c
c     ----------------------------------------------------------
c
      SUBROUTINE sweep1d( a, f, nx, s, e, b )
c
c     Perform a Jacobi sweep for a 1-d decomposition. use the 
c     previous values to update the value at each grid node of the 
c     subdomain.
c
      integer nx, s, e
      real*8 a(0:nx+1,s-1:e+1), f(0:nx+1,s-1:e+1),
     +                 b(0:nx+1,s-1:e+1)
c
      integer i, j
      real*8  h
c
      h = 1.0d0 / dble(nx+1)
      do 10 j=s, e
         do 10 i=1, nx
            b(i,j) = 0.25 * (a(i-1,j)+a(i,j+1)+a(i,j-1)+a(i+1,j)) - 
     +               h * h * f(i,j)
 10   continue

      return
      end
c
c     ----------------------------------------------------------
c
c    This subroutine uses stdandard send and recv mpi library calls 
c    for exchaning the boundary values. There are other communication 
c    calls which can also be used for better performace. 
c
      SUBROUTINE exchng1( a, nx, s, e, comm1d, nbrbottom, nbrtop )
c
       include "mpif.h"

       integer  nx, s, e
       real*8	 a(0:nx+1,s-1:e+1)

       integer  comm1d, nbrbottom, nbrtop
       integer  status(MPI_STATUS_SIZE), ierr

C     Send message to top processor
       CALL MPI_SEND( a(1,e), nx, MPI_DOUBLE_PRECISION, 
     $                 nbrtop, 0, comm1d, ierr )

C     Receive message from bottom
	CALL MPI_RECV( a(1,s-1), nx, MPI_DOUBLE_PRECISION, 
     $                 nbrbottom, 0, comm1d, status, ierr )

C     Send message to bottom processor
	CALL MPI_SEND( a(1,s), nx, MPI_DOUBLE_PRECISION, 
     $                nbrbottom, 1, comm1d, ierr )

C     Receive message from top processor
	CALL MPI_RECV( a(1,e+1), nx, MPI_DOUBLE_PRECISION, 
     $                nbrtop, 1, comm1d, status, ierr )


      return
      end
c
c    ----------------------------------------------------------
c
      DOUBLE PRECISION FUNCTION diff( a, b, nx, s, e )
c
c    ----------------------------------------------------------
c
c     This routine calculate the NORM of the vector, which is used for
c     checking the convergence of solution.

      integer 	nx, s, e
      real*8	a(0:nx+1, s-1:e+1), b(0:nx+1, s-1:e+1)

c     Local Variables
      real*8	sum
      integer 	i, j
c
      sum = 0.0d0
      do 10 j=s,e
         do 10 i=1,nx
            sum = sum + (a(i,j) - b(i,j)) ** 2
10    continue
c      
      diff = sum
      return
      end
c
c    ----------------------------------------------------------
c
c
